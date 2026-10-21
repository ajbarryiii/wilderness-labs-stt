#!/usr/bin/env bash
# Streamed exactness sample (plans/cpu-benchmark-mac.md M5): the 16 dev exactness utterances are never
# all resident on the Mac. For each batch: push the batch's capture directories (hash-checked, plus the
# group index), run the arm's `exactness` subcommand against the oracle digests (data/oracle/<model>/
# exactness, computed on NixOS by oracle-digests), then delete the batch from the Mac.
#
#   cpu1bit/shared/exactness-stream.sh AGENT ARM_BINARY EXPORT MODEL [THREADS] [COMPONENTS]
#   e.g. cpu1bit/shared/exactness-stream.sh H replay-b-neon MB1-derived MB1 8,1 2,3
# AGENT: the Mac workspace whose target/<AGENT>/release holds ARM_BINARY. EXPORT: data/exports/<EXPORT>.
# Results: Mac results/exactness/<ARM_BINARY>-<EXPORT>-exactness-b<k>.json; exit 1 on any mismatch.
set -euo pipefail
agent=$1 bin=$2 export_name=$3 model=$4 threads=${5:-8,1} comps=${6:-2,3}
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
py="$(cd -- "$here/../.." && pwd)/python"
src=/mnt/hd/wilderness-labs-stt/parakeet-binary/cpu1bit/captures/$model/exactness
D=$("$here/mac-root")
dest=stream/$model-$bin-$$
n=$(ls -d "$src"/[0-9][0-9] | wc -l)
batch=4
"$py" "$here/push.py" files "$src/index.json" --dest "$dest/exactness" >/dev/null
for ((b = 0; b * batch < n; b++)); do
  idx=() dirs=()
  for ((i = b * batch; i < (b + 1) * batch && i < n; i++)); do
    idx+=("$i"); dirs+=("$src/$(printf %02d "$i")")
  done
  "$py" "$here/push.py" files "${dirs[@]}" --dest "$dest/exactness" >/dev/null
  clips=$(IFS=,; echo "${idx[*]}")
  name="cpu1bit-$agent-xs-$bin-$export_name-b$b"
  job start "$name" --host mac --cwd "$D" --stall 30m --timeout 3h -- /usr/bin/perl "$D/ws/$agent/shared/mac/withlock" shared "$D/lock/active.lock" -- \
    "$D/target/$agent/release/$bin" exactness --export "$D/data/exports/$export_name" --captures "$D/data/$dest/exactness" \
    --reference "$D/data/oracle/$model/exactness" --clips "$clips" --threads "$threads" --components "$comps" \
    --out "$D/results/exactness/$bin-$export_name-exactness-b$b.json" >/dev/null
  status=0
  job wait "$name" --host mac --timeout 170m >/dev/null || status=$?
  "$py" "$here/push.py" rm --dest "$dest" >/dev/null
  if [[ $status -ne 0 ]]; then
    job tail "$name" --host mac -n 20; echo "batch $b FAILED"; exit 1
  fi
  echo "batch $b ($clips): exact"
  [[ $(( (b + 1) * batch )) -lt $n ]] && "$py" "$here/push.py" files "$src/index.json" --dest "$dest/exactness" >/dev/null
done
echo "exactness sample: all $n utterances exact for $bin on $export_name"
