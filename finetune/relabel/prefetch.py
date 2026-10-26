"""Download each teacher's pinned revision into relabel caches before scoring (no GPU, no lock).

    ./python prefetch.py [--teacher NAME ...]     (default: all three)

canary-qwen-2.5b also assembles its local directory (teachers.CanaryQwen.assemble); run it with
RELABEL_DEPS=canary-qwen-2.5b only if you also want to import NeMo, which prefetch does not need.
A gated repository whose terms the account has not accepted is reported as not scorable (exit 0);
this script never accepts terms or changes anything on the account.
"""
from __future__ import annotations

import argparse
import sys

import relabel_paths
import teachers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", nargs="+", choices=list(teachers.REGISTRY), default=list(teachers.REGISTRY))
    args = parser.parse_args(argv)
    relabel_paths.require_mount()
    for name in args.teacher:
        teacher = teachers.REGISTRY[name]()
        try:
            print(f"{name}: {teacher.prefetch()}", flush=True)
        except teachers.NotScorable as exc:
            print(f"{name}: NOT SCORABLE: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
