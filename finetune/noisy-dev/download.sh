#!/usr/bin/env bash
# Download the evaluation-pool sources of plans/build.md ("Evaluation-pool sources"), each into its
# own directory under /mnt/hd/wilderness-labs-stt/noisy-dev/raw/. Data only: nothing downloaded is
# executed or extracted here. One aria2c process (pinned, gcroots/aria2-bin): Zenodo throttles each
# connection to about 100-300 KB/s, so each file uses up to 8 connections, with 16 MiB/s aggregate
# for everything (plans/build.md "Running alongside the GPU pilot"). A watchdog suspends aria2c
# while the GPU pilot's data wait per step exceeds 0.05 s and resumes it below 0.02 s, or as soon as
# the pilot is not live (unit inactive or metrics older than 10 minutes). Resumable:
# rerun the same command (aria2c -c also continues the earlier sequential curl partials).
# Checksums are verified by the build (noisy_sources.verify), not here; <name>.done markers, written
# by aria2-done.sh when a file completes, are resume hints only. Writes raw/SHA256SUMS at the end.
#   ../parakeet-ternary/heavy noisy-download --mem-max 1G -- "$PWD/download.sh"
set -euo pipefail
mountpoint -q /mnt/hd || { echo '/mnt/hd is not mounted; refusing artifact writes.' >&2; exit 1; }
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
art=/mnt/hd/wilderness-labs-stt/noisy-dev
raw=$art/raw
aria2c=$art/gcroots/aria2-bin/bin/aria2c  # nixpkgs aria2 1.37.0
zenodo=https://zenodo.org/api/records
list=$art/tmp/aria2-input.txt
mkdir -p "$raw" "$art/tmp"

entries=()
add() {  # add SOURCE_DIR URL FILENAME
  mkdir -p "$raw/$1"
  [[ -f "$raw/$1/$3.done" ]] || entries+=("$2"$'\n'"  dir=$raw/$1"$'\n'"  out=$3")
}
for f in FSD50K.eval_audio.zip FSD50K.eval_audio.z01 FSD50K.metadata.zip FSD50K.ground_truth.zip FSD50K.doc.zip; do
  add fsd50k "$zenodo/4060432/files/$f/content" "$f"
done
for env in DKITCHEN DLIVING DWASHING NFIELD NPARK NRIVER OHALLWAY OMEETING OOFFICE PCAFETER PRESTO \
           PSTATION SPSQUARE STRAFFIC TBUS TCAR TMETRO; do
  add demand "$zenodo/1227121/files/${env}_16k.zip/content" "${env}_16k.zip"
done
add demand "$zenodo/1227121/files/DEMAND.pdf/content" DEMAND.pdf
for f in readme.txt mic1raw.wav mic2raw16b.wav mic3raw.wav mic4raw.wav; do
  add gunshots-ipod "$zenodo/3997406/files/$f/content" "$f"
done
add but-reverbdb https://merlin.fit.vutbr.cz/ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz BUT_ReverbDB_rel_19_06_RIR-Only.tgz
add musan https://www.openslr.org/resources/17/musan.tar.gz musan.tar.gz

if (( ${#entries[@]} )); then
  printf '%s\n' "${entries[@]}" > "$list"
  echo "[download] ${#entries[@]} files to fetch" >&2
  "$aria2c" --input-file="$list" --continue=true --max-concurrent-downloads=6 \
    --max-connection-per-server=8 --split=8 --min-split-size=64M \
    --max-overall-download-limit=16M --max-tries=0 --retry-wait=30 --timeout=120 \
    --auto-file-renaming=false --allow-overwrite=false --file-allocation=none \
    --on-download-complete="$here/aria2-done.sh" --summary-interval=300 \
    --console-log-level=warn --show-console-readout=false &
  aria2_pid=$!
  trap 'kill -CONT "$aria2_pid" 2>/dev/null || true' EXIT
  ( paused=0
    while kill -0 "$aria2_pid" 2>/dev/null; do
      sleep 60
      # Protect the pilot only while it is live: its unit is active and its newest metrics line is
      # under 10 minutes old. Otherwise (pilot finished, crashed, or between arms) never stay paused.
      wait=""
      m=$(ls -t /mnt/hd/wilderness-labs-stt/parakeet-binary/runs/pilot-*/metrics.jsonl 2>/dev/null | head -1 || true)
      if [[ -n "$m" ]] && systemctl --user -q is-active parakeet-binary-binary-pilot \
         && (( $(date +%s) - $(stat -c %Y "$m") < 600 )); then
        wait=$(tail -n 50 "$m" | grep -o '"data_wait_s_per_step": [0-9.e-]*' | tail -1 | grep -o '[0-9.e-]*$' || true)
      fi
      if [[ -z "$wait" ]]; then
        if (( paused )); then echo "[download] pilot not live: resuming transfers" >&2; kill -CONT "$aria2_pid"; paused=0; fi
        continue
      fi
      if (( ! paused )) && awk -v w="$wait" 'BEGIN { exit !(w > 0.05) }'; then
        echo "[download] pilot data wait ${wait}s/step: pausing transfers" >&2; kill -STOP "$aria2_pid"; paused=1
      elif (( paused )) && awk -v w="$wait" 'BEGIN { exit !(w < 0.02) }'; then
        echo "[download] pilot data wait ${wait}s/step: resuming transfers" >&2; kill -CONT "$aria2_pid"; paused=0
      fi
    done ) &
  wait "$aria2_pid" || { echo "[download] aria2c failed; rerun to resume" >&2; exit 1; }
fi

(cd "$raw" && find . -type f ! -name '*.done' ! -name '*.aria2' ! -name SHA256SUMS -print0 | sort -z \
   | xargs -0 nice -n 19 ionice -c 3 sha256sum > SHA256SUMS)  # idle I/O class: never contends with the pilot
echo "[download] done; $(wc -l < "$raw/SHA256SUMS") files, $(du -sh "$raw" | cut -f1)" >&2
