#!/usr/bin/env bash
# aria2c --on-download-complete hook for download.sh: args are GID, file count, first file path.
# Writes the <name>.done resume hint next to the completed file.
set -euo pipefail
[[ "${2:-0}" -ge 1 && -f "${3:-}" ]] && touch "$3.done"
