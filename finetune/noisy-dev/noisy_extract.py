"""Safe extraction of untrusted archives (plans/build.md "Evaluation-pool sources").

Downloads are untrusted data. Archives are only parsed here, never executed or handed to an
external extractor, and members are written by this code alone:

- every member name is checked before anything is written (UnsafeArchive on an absolute path, a
  '..' component, a backslash, a drive letter, a NUL or an empty name), and selected names may not
  repeat;
- only regular files are written; a selected link, device or other special member is refused;
- members go to a new staging directory next to the destination, which is renamed into place only
  when every selected member was written and verified (zip: CRC-32 and size), so a refused or
  failed extraction leaves nothing at the destination.

Zip archives are read by an own parser of the central directory (zip64 included) so that split
archives (FSD50K's .z01 + .zip) work: each entry's local header is found at its disk's start
plus its offset in the concatenation of the segments. Stored and deflated members only.

Budgets (review round 2, finding 5): every extraction runs under a finite Budget, checked before
anything is written and again while streaming: the archive's member count (all members, selected
or not), each member's expanded size and the total expanded size. Inflation is bounded per call
(OUT_CHUNK bytes of output per zlib call), and a member producing more than it declares is refused,
so a decompression bomb costs at most one chunk beyond its budget. A deflated member must be one
complete deflate stream that consumes exactly its declared compressed size: data after the
stream's end is refused as soon as it appears (eof, unused_data and unconsumed_tail are checked
after every call), and a stream without an end is refused as truncated (review round 4).

Metadata budgets (review round 3, finding 1): archive metadata is bounded before it is read. A
zip's central-directory size and entry count, classic or ZIP64, are checked against the budget
before the directory is read (only the fixed-size end records are read first). Tar archives are
parsed by an own reader (parse_tar) instead of tarfile, whose stream parser reads PAX and GNU
extended headers whole before any check: every 512-byte header is checksum-verified, an extended
header ('x', 'g', 'L', 'K') larger than max_metadata_bytes is refused before its data is read,
extended headers count as members, their total is capped (METADATA_TOTAL_FACTOR x
max_metadata_bytes), and the decompressed stream as a whole may not exceed the bytes the budget
allows (members, metadata, headers and padding).
"""
from __future__ import annotations

import os
import shutil
import stat
import dataclasses
import gzip
import struct
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath

CHUNK = 1 << 20       # compressed bytes read per step
OUT_CHUNK = 1 << 20   # most decompressed bytes one zlib call may produce


@dataclasses.dataclass(frozen=True)
class Budget:
    max_members: int          # every member of the archive, directories and unselected ones included
    max_member_bytes: int     # expanded size of one member
    max_total_bytes: int      # expanded size of all members considered (zip: selected; tar: every regular member)
    max_metadata_bytes: int = 1 << 20  # a zip central directory; one tar extended header


METADATA_TOTAL_FACTOR = 64  # tar: all extended headers together <= 64 x max_metadata_bytes
DEFAULT_BUDGET = Budget(max_members=10_000, max_member_bytes=256 << 20, max_total_bytes=1 << 30)


class UnsafeArchive(ValueError):
    """The archive holds a member that may not be extracted."""


def safe_member_name(name: str) -> PurePosixPath:
    """The member's relative POSIX path, or UnsafeArchive."""
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise UnsafeArchive(f"unsafe member name {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or (path.parts and ":" in path.parts[0]):
        raise UnsafeArchive(f"absolute member name {name!r}")
    parts = [p for p in path.parts if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        raise UnsafeArchive(f"unsafe member name {name!r}")
    return PurePosixPath(*parts)


def _staging(dest: Path) -> Path:
    staging = dest.with_name(f".staging-{dest.name}-{os.getpid()}")
    if dest.exists():
        raise FileExistsError(f"{dest} exists; extraction writes only to a new directory")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    return staging


def _target(staging: Path, rel: PurePosixPath) -> Path:
    out = staging.joinpath(*rel.parts)
    if staging.resolve() not in out.resolve().parents:  # belt and braces after the name check
        raise UnsafeArchive(f"{rel} escapes the destination")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------------------------- zip

class _Segments:
    """Read access to the concatenation of a split archive's segments (disk 0 first)."""

    def __init__(self, segments: Sequence[Path]) -> None:
        self.paths = [Path(p) for p in segments]
        self.sizes = [p.stat().st_size for p in self.paths]
        self.base = [sum(self.sizes[:k]) for k in range(len(self.sizes))]
        self.handles = [open(p, "rb") for p in self.paths]

    def close(self) -> None:
        for h in self.handles:
            h.close()

    def disk_offset(self, disk: int, offset: int) -> int:
        if not 0 <= disk < len(self.paths):
            raise UnsafeArchive(f"entry on disk {disk}, archive has {len(self.paths)} segments")
        return self.base[disk] + offset

    def read(self, pos: int, n: int) -> bytes:
        out = bytearray()
        while n > 0:
            k = max(i for i, b in enumerate(self.base) if b <= pos)
            local = pos - self.base[k]
            if local >= self.sizes[k]:
                raise UnsafeArchive("read past the end of the archive")
            h = self.handles[k]
            h.seek(local)
            chunk = h.read(min(n, self.sizes[k] - local))
            out += chunk
            pos += len(chunk)
            n -= len(chunk)
        return bytes(out)


def _central_directory(seg: _Segments, budget: Budget) -> list[dict]:
    last = len(seg.paths) - 1
    tail_len = min(seg.sizes[last], 65557)
    tail = seg.read(seg.base[last] + seg.sizes[last] - tail_len, tail_len)
    at = tail.rfind(b"PK\x05\x06")
    if at < 0 or len(tail) - at < 22:
        raise UnsafeArchive("no end-of-central-directory record")
    (_, _disk, cd_disk, _, entries, cd_size, cd_offset, _) = struct.unpack("<IHHHHIIH", tail[at:at + 22])
    eocd_pos = seg.base[last] + seg.sizes[last] - tail_len + at
    if 0xFFFF in (cd_disk, entries) or 0xFFFFFFFF in (cd_size, cd_offset):
        loc = seg.read(eocd_pos - 20, 20)
        sig, z64_disk, z64_off, _total = struct.unpack("<IIQI", loc)
        if sig != 0x07064B50:
            raise UnsafeArchive("zip64 locator missing")
        rec = seg.read(seg.disk_offset(z64_disk, z64_off), 56)
        (sig, _, _, _, _, cd_disk, _, entries, cd_size, cd_offset) = struct.unpack("<IQHHIIQQQQ", rec)
        if sig != 0x06064B50:
            raise UnsafeArchive("zip64 end record missing")
    # Bound the metadata before reading it.
    if entries > budget.max_members:
        raise UnsafeArchive(f"{entries} members exceed the budget of {budget.max_members}")
    if cd_size > budget.max_metadata_bytes:
        raise UnsafeArchive(f"central directory of {cd_size} bytes exceeds the metadata budget "
                            f"of {budget.max_metadata_bytes}")
    if cd_size > sum(seg.sizes) or cd_size < 46 * entries:
        raise UnsafeArchive(f"central directory size {cd_size} is inconsistent with the archive")
    data = seg.read(seg.disk_offset(cd_disk, cd_offset), cd_size)
    out, pos = [], 0
    for _ in range(entries):
        if data[pos:pos + 4] != b"PK\x01\x02" or pos + 46 > len(data):
            raise UnsafeArchive("bad central directory entry")
        (_, made, _, flags, method, _, _, crc, csize, usize, nlen, elen, clen, disk, _, ext,
         offset) = struct.unpack("<IHHHHHHIIIHHHHHII", data[pos:pos + 46])
        name = data[pos + 46:pos + 46 + nlen].decode("utf-8" if flags & 0x800 else "cp437")
        extra = data[pos + 46 + nlen:pos + 46 + nlen + elen]
        pos += 46 + nlen + elen + clen
        # zip64 extended information: only the fields saturated in the fixed header, in order.
        e = 0
        while e + 4 <= len(extra):
            tag, size = struct.unpack("<HH", extra[e:e + 4])
            if tag == 0x0001:
                body, b = extra[e + 4:e + 4 + size], 0
                if usize == 0xFFFFFFFF:
                    usize, b = struct.unpack("<Q", body[b:b + 8])[0], b + 8
                if csize == 0xFFFFFFFF:
                    csize, b = struct.unpack("<Q", body[b:b + 8])[0], b + 8
                if offset == 0xFFFFFFFF:
                    offset, b = struct.unpack("<Q", body[b:b + 8])[0], b + 8
                if disk == 0xFFFF:
                    disk = struct.unpack("<I", body[b:b + 4])[0]
            e += 4 + size
        out.append({"name": name, "flags": flags, "method": method, "crc": crc, "csize": csize, "usize": usize,
                    "disk": disk, "offset": offset, "mode": ext >> 16, "unix": made >> 8 == 3})
    return out


def _check_zip_entry(entry: dict) -> bool:
    """True for a regular file to consider, False for a directory; UnsafeArchive otherwise."""
    if entry["name"].endswith("/"):
        return False
    kind = stat.S_IFMT(entry["mode"]) if entry["unix"] else 0
    if kind not in (0, stat.S_IFREG):
        raise UnsafeArchive(f"{entry['name']}: not a regular file (mode {entry['mode']:o})")
    if entry["flags"] & 0x1:
        raise UnsafeArchive(f"{entry['name']}: encrypted")
    if entry["method"] not in (0, 8):
        raise UnsafeArchive(f"{entry['name']}: compression method {entry['method']}")
    return True


def _central_entry_count(seg: _Segments) -> int:
    last = len(seg.paths) - 1
    tail_len = min(seg.sizes[last], 65557)
    tail = seg.read(seg.base[last] + seg.sizes[last] - tail_len, tail_len)
    at = tail.rfind(b"PK\x05\x06")
    if at < 0 or len(tail) - at < 22:
        raise UnsafeArchive("no end-of-central-directory record")
    return struct.unpack("<H", tail[at + 10:at + 12])[0]


class _Sink:
    """Writes a member's data, keeping its CRC-32 and size; refuses more than the declared size."""

    def __init__(self, handle, declared: int, name: str) -> None:
        self.handle, self.declared, self.name = handle, declared, name
        self.crc = self.size = 0

    def write(self, data: bytes) -> None:
        self.crc, self.size = zlib.crc32(data, self.crc), self.size + len(data)
        if self.size > self.declared:
            raise UnsafeArchive(f"{self.name}: inflates to more data than declared ({self.declared} bytes)")
        self.handle.write(data)


def extract_zip(segments: Sequence[Path], dest: Path, select: Callable[[str], bool],
                budget: Budget = DEFAULT_BUDGET) -> list[dict]:
    """Extract the selected regular members of a zip archive (its segments in disk order, one for
    an ordinary archive) into the new directory dest, within the budget; returns [{name, size,
    crc32}] sorted."""
    dest = Path(dest)
    seg = _Segments(segments)
    staging = None
    try:
        count = _central_entry_count(seg)
        if count != 0xFFFF and count > budget.max_members:
            raise UnsafeArchive(f"{count} members exceed the budget of {budget.max_members}")
        entries = _central_directory(seg, budget)
        if len(entries) > budget.max_members:
            raise UnsafeArchive(f"{len(entries)} members exceed the budget of {budget.max_members}")
        chosen, seen, total = [], set(), 0
        for entry in entries:
            name = entry["name"]
            if (name.endswith("/") and name.strip("/") in ("", ".") and entry["usize"] == 0
                    and (entry["csize"] == 0 or (entry["method"] == 8 and entry["csize"] <= 2))):
                continue  # "./": the archive root, no data; stored, or deflate's 2-byte empty stream (counted above)
            rel = safe_member_name(name)
            if not _check_zip_entry(entry) or not select(str(rel)):
                continue
            if str(rel) in seen:
                raise UnsafeArchive(f"duplicate member {rel}")
            if entry["usize"] > budget.max_member_bytes:
                raise UnsafeArchive(f"{rel}: {entry['usize']} bytes exceed the member budget")
            if entry["method"] == 0 and entry["csize"] != entry["usize"]:
                raise UnsafeArchive(f"{rel}: stored member with different sizes")
            total += entry["usize"]
            if total > budget.max_total_bytes:
                raise UnsafeArchive(f"selected members exceed the total budget of {budget.max_total_bytes} bytes")
            seen.add(str(rel))
            chosen.append((rel, entry))
        staging = _staging(dest)
        records = []
        for rel, entry in chosen:
            pos = seg.disk_offset(entry["disk"], entry["offset"])
            header = seg.read(pos, 30)
            if header[:4] != b"PK\x03\x04":
                raise UnsafeArchive(f"{rel}: bad local header")
            nlen, elen = struct.unpack("<HH", header[26:30])
            pos += 30 + nlen + elen
            inflate = zlib.decompressobj(-15) if entry["method"] == 8 else None
            left = entry["csize"]
            with open(_target(staging, rel), "wb") as handle:
                sink = _Sink(handle, entry["usize"], str(rel))
                while left > 0:
                    if inflate is not None and inflate.eof:
                        raise UnsafeArchive(f"{rel}: trailing data after the deflate stream ({left} bytes)")
                    chunk = seg.read(pos, min(CHUNK, left))
                    pos, left = pos + len(chunk), left - len(chunk)
                    if inflate is None:
                        sink.write(chunk)
                        continue
                    while chunk:  # at most OUT_CHUNK bytes of output per call
                        sink.write(inflate.decompress(chunk, OUT_CHUNK))
                        if inflate.unused_data or (inflate.eof and inflate.unconsumed_tail):
                            raise UnsafeArchive(f"{rel}: trailing data after the deflate stream")
                        chunk = inflate.unconsumed_tail
                if inflate is not None and not inflate.eof:
                    raise UnsafeArchive(f"{rel}: truncated deflate stream (no end within the declared "
                                        f"{entry['csize']} compressed bytes)")
            crc, size = sink.crc, sink.size
            if crc != entry["crc"] or size != entry["usize"]:
                raise UnsafeArchive(f"{rel}: CRC-32 or size mismatch")
            records.append({"name": str(rel), "size": size, "crc32": f"{crc:08x}"})
        os.replace(staging, dest)
        staging = None
        return sorted(records, key=lambda r: r["name"])
    finally:
        seg.close()
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


# --------------------------------------------------------------------------------------------- tar

class _TarStream:
    """Exact, bounded reads from the decompressed stream of a gzip tar archive (the only format accepted)."""

    def __init__(self, archive: Path, limit: int) -> None:
        self.file = open(archive, "rb")
        magic = self.file.read(6)
        self.file.seek(0)
        # Only the compression the registered sources use (MUSAN, BUT: gzip). XZ (whose header can
        # demand a huge decoder dictionary), bzip2, plain tar and anything else are refused by
        # magic, whatever the file name (review round 5).
        if magic[:3] != b"\x1f\x8b\x08":
            self.file.close()
            raise UnsafeArchive(f"{archive}: unsupported compression (only gzip/deflate tar archives are "
                                f"accepted; magic {magic.hex()})")
        self.raw = gzip.GzipFile(fileobj=self.file)
        self.limit, self.consumed = limit, 0

    def close(self) -> None:
        self.raw.close()
        self.file.close()

    def _take(self, n: int) -> None:
        self.consumed += n
        if self.consumed > self.limit:
            raise UnsafeArchive(f"decompressed tar stream exceeds its budget of {self.limit} bytes")

    def block(self) -> bytes | None:
        """The next 512-byte block, or None at a clean end of stream."""
        self._take(512)
        data = self.raw.read(512)
        while data and len(data) < 512:
            more = self.raw.read(512 - len(data))
            if not more:
                break
            data += more
        if not data:
            return None
        if len(data) != 512:
            raise UnsafeArchive("truncated tar header")
        return data

    def read(self, n: int) -> bytes:
        self._take(n)
        out = bytearray()
        while len(out) < n:
            chunk = self.raw.read(min(CHUNK, n - len(out)))
            if not chunk:
                raise UnsafeArchive("truncated tar member")
            out += chunk
        return bytes(out)

    def copy(self, n: int, write) -> None:
        left = n
        while left > 0:
            chunk = self.read(min(CHUNK, left))
            write(chunk)
            left -= len(chunk)


def _tar_number(field: bytes) -> int:
    if field and field[0] & 0x80:  # GNU base-256
        if field[0] != 0x80:
            raise UnsafeArchive("negative base-256 number in a tar header")
        return int.from_bytes(field[1:], "big")
    text = field.split(b"\0", 1)[0].strip()
    try:
        return int(text or b"0", 8)
    except ValueError:
        raise UnsafeArchive(f"bad number {field!r} in a tar header") from None


def _tar_text(field: bytes) -> str:
    try:
        return field.split(b"\0", 1)[0].decode("utf-8")
    except UnicodeDecodeError:
        raise UnsafeArchive("tar header field is not UTF-8") from None


def _pax_records(data: bytes) -> dict[str, str]:
    """Records 'LEN key=value\\n' of a PAX extended header (validated)."""
    out, pos = {}, 0
    while pos < len(data):
        if not data[pos:].strip(b"\0"):
            break
        space = data.find(b" ", pos)
        if space < 0:
            raise UnsafeArchive("malformed PAX record")
        try:
            length = int(data[pos:space])
        except ValueError:
            raise UnsafeArchive("malformed PAX record length") from None
        record = data[space + 1:pos + length]
        if length <= 0 or pos + length > len(data) or not record.endswith(b"\n") or b"=" not in record:
            raise UnsafeArchive("malformed PAX record")
        key, value = record[:-1].split(b"=", 1)
        try:
            out[key.decode("utf-8")] = value.decode("utf-8")
        except UnicodeDecodeError:
            raise UnsafeArchive("PAX record is not UTF-8") from None
        pos += length
    return out


def _round512(n: int) -> int:
    return (n + 511) // 512 * 512


def parse_tar(archive: Path, budget: Budget):
    """Yield (name, typeflag, size, stream) for each member of a tar archive, after its header
    (and any extended headers before it) passed the budget checks; the consumer must read or skip
    exactly `size` bytes of `stream` (extract_tar does). Extended headers are applied, not yielded."""
    metadata_total_limit = METADATA_TOTAL_FACTOR * budget.max_metadata_bytes
    limit = budget.max_total_bytes + metadata_total_limit + 1024 * (budget.max_members + 2)
    stream = _TarStream(archive, limit)
    try:
        members = metadata_total = 0
        pax: dict[str, str] = {}
        long_name: str | None = None
        while True:
            header = stream.block()
            if header is None or header == bytes(512):
                return  # end of archive (zero block or end of stream)
            stored = _tar_number(header[148:156])
            unsigned = sum(header[:148]) + 8 * 32 + sum(header[156:])
            signed = sum(b - 256 if b > 127 else b for b in header[:148]) + 8 * 32 + \
                sum(b - 256 if b > 127 else b for b in header[156:])
            if stored not in (unsigned, signed):
                raise UnsafeArchive("tar header checksum mismatch")
            members += 1
            if members > budget.max_members:
                raise UnsafeArchive(f"more than {budget.max_members} members (extended headers included)")
            typeflag = header[156:157]
            size = _tar_number(header[124:136])
            if typeflag in (b"x", b"g", b"L", b"K"):
                if size > budget.max_metadata_bytes:
                    raise UnsafeArchive(f"extended header of {size} bytes exceeds the metadata budget "
                                        f"of {budget.max_metadata_bytes}")
                metadata_total += size
                if metadata_total > metadata_total_limit:
                    raise UnsafeArchive("extended headers exceed the total metadata budget")
                data = stream.read(_round512(size))[:size]
                if typeflag == b"x":
                    pax = _pax_records(data)
                elif typeflag == b"g":
                    _pax_records(data)  # validated; global records are not applied
                elif typeflag == b"L":
                    long_name = data.split(b"\0", 1)[0].decode("utf-8", errors="strict")
                continue
            name = _tar_text(header[0:100])
            if header[257:262] == b"ustar":
                prefix = _tar_text(header[345:500])
                if prefix:
                    name = f"{prefix}/{name}"
            if long_name is not None:
                name = long_name
            if "path" in pax:
                name = pax["path"]
            if "size" in pax:
                try:
                    size = int(pax["size"])
                except ValueError:
                    raise UnsafeArchive("bad PAX size") from None
            pax, long_name = {}, None
            if size < 0:
                raise UnsafeArchive("negative member size")
            yield name, typeflag, size, stream
    finally:
        stream.close()


def extract_tar(archive: Path, dest: Path, select: Callable[[str], bool],
                budget: Budget = DEFAULT_BUDGET) -> list[dict]:
    """Extract the selected regular members of a (compressed) tar archive, read as a stream by
    parse_tar, into the new directory dest, within the budget (members and sizes count whether
    selected or not: the stream decompresses all of them); returns [{name, size}] sorted.
    Unselected links are skipped."""
    dest = Path(dest)
    staging = _staging(dest)
    try:
        records, seen, total = [], set(), 0
        for name, typeflag, size, stream in parse_tar(archive, budget):
            regular = typeflag in (b"0", b"\0", b"7")
            if regular:
                if size > budget.max_member_bytes:
                    raise UnsafeArchive(f"{name!r}: {size} bytes exceed the member budget")
                total += size
                if total > budget.max_total_bytes:
                    raise UnsafeArchive(f"members exceed the total budget of {budget.max_total_bytes} bytes")
            if typeflag == b"5" and name.strip("/") in ("", "."):  # "./": the archive root, no data
                stream.copy(_round512(size), lambda chunk: None)
                continue
            rel = safe_member_name(name)
            if typeflag == b"5" or not select(str(rel)):
                stream.copy(_round512(size), lambda chunk: None)  # skip data and padding
                continue
            if not regular:
                raise UnsafeArchive(f"{rel}: not a regular file (type {typeflag!r})")
            if str(rel) in seen:
                raise UnsafeArchive(f"duplicate member {rel}")
            seen.add(str(rel))
            with open(_target(staging, rel), "wb") as out:
                stream.copy(size, out.write)
            stream.copy(_round512(size) - size, lambda chunk: None)
            records.append({"name": str(rel), "size": size})
        os.replace(staging, dest)
        staging = None
        return sorted(records, key=lambda r: r["name"])
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
