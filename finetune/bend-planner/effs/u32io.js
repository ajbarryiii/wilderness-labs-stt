// u32io (see u32io.c): the JS lane keeps an Array<U32> as a JS array of numbers, so the words go
// through a buffer and are viewed as little-endian u32. Same answers as the C lane: a short read
// fails with EIO (5), n past the array's length with EINVAL (22).

function u32s_read(file, offset, n, a) {
  const fs = require("fs");
  if (n > a.length) {
    return io_tup(file, io_tup(a, io_fail(22)));
  }
  const len = n * 4;
  const b = Buffer.alloc(Math.max(len, 4));
  let got = 0;
  try {
    while (got < len) {
      const k = fs.readSync(file, b, got, len - got, offset + got);
      if (k === 0) {
        return io_tup(file, io_tup(a, io_fail(5)));
      }
      got += k;
    }
  } catch (e) {
    return io_tup(file, io_tup(a, io_fail(Math.abs(e.errno ?? 5))));
  }
  for (let i = 0; i < n; i++) {
    a[i] = b.readUInt32LE(4 * i);
  }
  return io_tup(file, io_tup(a, io_done(n)));
}

function u32s_write(file, n, a) {
  const fs = require("fs");
  if (n > a.length) {
    return io_tup(file, io_tup(a, io_fail(22)));
  }
  const b = Buffer.alloc(n * 4);
  for (let i = 0; i < n; i++) {
    b.writeUInt32LE(a[i] >>> 0, 4 * i);
  }
  let at = 0;
  try {
    while (at < b.length) {
      const k = fs.writeSync(file, b, at, b.length - at, null);
      if (k === 0) {
        return io_tup(file, io_tup(a, io_fail(5)));
      }
      at += k;
    }
  } catch (e) {
    return io_tup(file, io_tup(a, io_fail(Math.abs(e.errno ?? 5))));
  }
  return io_tup(file, io_tup(a, io_done({ $: CID(Unit) })));
}

io_eff(CID(U32s.read), u32s_read);
io_eff(CID(U32s.write), u32s_write);
