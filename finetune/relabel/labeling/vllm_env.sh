#!/usr/bin/env bash
# The isolated vLLM environment of the labeling pass (labeling plan Part C "Engine").
#
#   labeling/vllm_env.sh install     create $RELABEL_ROOT/deps/vllm-$VLLM_VERSION (a venv) and pip-install the
#                                    pinned vLLM into it (network; run as a job); records pip freeze
#   labeling/vllm_env.sh python ...  run relabel_boot.py with that venv's interpreter (like ./python)
#
# vLLM pins its own torch, which differs from the shared runtime's, so it cannot be an appended RELABEL_DEPS
# directory: it gets its own venv, built from the shared runtime's base interpreter without its site-packages.
# The default environment (./python, the shared runtime) is not changed. Native libraries (libstdc++, the
# driver) come from the shared runtime's LD_LIBRARY_PATH, read at run time. Roots: private/local.env.
set -euo pipefail
VLLM_VERSION=0.32.0
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="$here/private/local.env"
if [[ -f "$env_file" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^([A-Z_][A-Z0-9_]*)=(.*)$ ]] || continue
    key="${BASH_REMATCH[1]}"; value="${BASH_REMATCH[2]}"
    [[ -n "${!key:-}" ]] || export "$key=$value"
  done < "$env_file"
fi
for key in DATA_MOUNT RELABEL_ROOT PARAKEET_TERNARY_ROOT PARAKEET_BINARY_ROOT SHARED_RUNTIME; do
  [[ -n "${!key:-}" ]] || { echo "vllm_env: $key is not set (environment or $env_file)" >&2; exit 2; }
done
mountpoint -q "$DATA_MOUNT" || { echo "vllm_env: $DATA_MOUNT is not mounted; refusing artifact writes." >&2; exit 1; }
venv="$RELABEL_ROOT/deps/vllm-$VLLM_VERSION"
runtime_ld="$("$SHARED_RUNTIME" -c 'import os; print(os.environ.get("LD_LIBRARY_PATH", ""))')"

case "${1:-}" in
  install)
    base="$(env -u PYTHONPATH "$SHARED_RUNTIME" -I -c 'import sys; print(sys.executable)')"
    mkdir -p "$RELABEL_ROOT/caches/pip" "$RELABEL_ROOT/tmp"
    export PIP_CACHE_DIR="$RELABEL_ROOT/caches/pip" TMPDIR="$RELABEL_ROOT/tmp" LD_LIBRARY_PATH="$runtime_ld"
    unset PYTHONPATH
    [[ -x "$venv/bin/python" ]] || "$base" -m venv --without-pip "$venv"
    "$venv/bin/python" -I -m ensurepip --upgrade
    # transformers pinned to the shared runtime's (stage 1's) version, so the processor, feature extractor and
    # chat template are stage 1's; pyarrow and soundfile read C1's shards.
    "$venv/bin/python" -I -m pip install --no-input "vllm[audio]==$VLLM_VERSION" "transformers==5.17.0" \
      "pyarrow==25.0.1" "soundfile==0.13.1"
    "$venv/bin/python" -I -m pip freeze > "$venv/FREEZE.txt"
    "$venv/bin/python" -I -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, torch.version.cuda)' \
      | tee "$venv/INSTALLED.txt" ;;
  python)
    shift
    [[ -x "$venv/bin/python" ]] || { echo "vllm_env: $venv is missing (run: labeling/vllm_env.sh install)" >&2; exit 2; }
    # vLLM forks its EngineCore by default. The parent has already run torch CPU ops on its OpenMP (libgomp)
    # thread pool by then, and libgomp is not fork-safe: the forked engine segfaulted in gomp_team_start on its
    # first parallel CPU op (2026-10-22 smoke). A spawned engine starts a fresh interpreter instead.
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    # FlashInfer kernels are unavailable here (no nvcc for its JIT), but vLLM 0.32's sampler check still picks
    # FlashInfer's top-k/top-p sampler and fails in the engine warmup. Decoding is greedy; use vLLM's own sampler.
    export VLLM_USE_FLASHINFER_SAMPLER=0
    # Triton (torch.compile in the engine) finds libcuda with `/sbin/ldconfig -p`, which NixOS does not have;
    # point it at the driver directory on the runtime's library path instead.
    if [[ -z "${TRITON_LIBCUDA_PATH:-}" ]]; then
      IFS=: read -ra ld_dirs <<< "$runtime_ld"
      for d in "${ld_dirs[@]}"; do
        if [[ -n "$d" && -e "$d/libcuda.so.1" ]]; then export TRITON_LIBCUDA_PATH="$d"; break; fi
      done
      [[ -n "${TRITON_LIBCUDA_PATH:-}" ]] || { echo "vllm_env: no libcuda.so.1 on the runtime's LD_LIBRARY_PATH" >&2; exit 2; }
    fi
    # Triton's bundled CUDA tools (ptxas, ptxas-blackwell, ...) are generic-Linux executables that NixOS cannot
    # start (no /lib64/ld-linux-x86-64.so.2). Run each through the glibc loader the venv's Python itself uses,
    # via a wrapper under the venv, and point Triton at the wrappers (TRITON_<TOOL>_PATH).
    tool_dir="$(cd -- "$venv"/lib/python3*/site-packages/triton/backends/nvidia/bin && pwd)"
    loader="$("$venv/bin/python" -I -c 'print(next(l.split()[-1] for l in open("/proc/self/maps") if l.rstrip().endswith("/ld-linux-x86-64.so.2")))')"
    wrap_dir="$venv/nixos-bin"
    mkdir -p "$wrap_dir"
    for tool in ptxas ptxas-blackwell cuobjdump nvdisasm; do
      [[ -x "$tool_dir/$tool" ]] || continue
      body="#!/usr/bin/env bash
exec \"$loader\" --library-path \"$(dirname -- "$loader"):$runtime_ld\" \"$tool_dir/$tool\" \"\$@\""
      if [[ ! -x "$wrap_dir/$tool" || "$(cat "$wrap_dir/$tool")" != "$body" ]]; then
        printf '%s\n' "$body" > "$wrap_dir/$tool.tmp$$" && chmod +x "$wrap_dir/$tool.tmp$$" && mv -f "$wrap_dir/$tool.tmp$$" "$wrap_dir/$tool"
      fi
      key="TRITON_$(tr 'a-z-' 'A-Z_' <<< "$tool")_PATH"
      [[ -n "${!key:-}" ]] || export "$key=$wrap_dir/$tool"
    done
    export LD_LIBRARY_PATH="$runtime_ld" RELABEL_VLLM_VERSION="$VLLM_VERSION"
    unset PYTHONPATH RELABEL_DEPS
    exec "$venv/bin/python" "$here/relabel_boot.py" "$@" ;;
  *)
    sed -n '2,12p' "$0" >&2; exit 2 ;;
esac
