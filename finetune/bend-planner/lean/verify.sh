#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
lake build
lake env lean Audit.lean
"${PYTHON:-python3}" fidelity.py "$@"
"${PYTHON:-python3}" conversion_probes.py
