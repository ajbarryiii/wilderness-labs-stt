#!/usr/bin/env bash
# Build and run the x86 golden-vector generator against the read-only LocalFlow snapshot.
#   job start cpu1bit-golden --needs cpu-heavy -- cpu1bit/shared/golden/run.sh
# Output: /mnt/hd/wilderness-labs-stt/parakeet-binary/cpu1bit/golden/<UTC stamp>/ (+ `latest` symlink).
set -euo pipefail
mountpoint -q /mnt/hd || { echo "/mnt/hd not mounted" >&2; exit 1; }
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
base=/mnt/hd/wilderness-labs-stt/parakeet-binary/cpu1bit
export CARGO_TARGET_DIR="$base/golden-target"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
out="$base/golden/$stamp"
cargo build --offline --release --manifest-path "$here/Cargo.toml"
"$CARGO_TARGET_DIR/release/cpu1bit-golden-x86" "$out"
cp "$here/src/main.rs" "$out/generator-main.rs"
sha256sum "$out"/* > "$out/SHA256SUMS"
ln -sfn "$stamp" "$base/golden/latest"
echo "golden vectors: $out"
