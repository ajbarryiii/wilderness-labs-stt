"""Load B's work-in-progress x1/x2 modules (kept outside gpu1bit/int8_mma/ while H's selection freeze is in
force) under their final package names, gpu1bit.int8_mma.<name>, so their code is exactly what will be
moved into int8_mma/ later. Usage: `import wip` (this directory on sys.path) before importing them."""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = ("kernels_parts", "layer_parts", "parts_run", "test_parts")


def load(name: str):
    full = f"gpu1bit.int8_mma.{name}"
    if full in sys.modules:
        return sys.modules[full]
    import gpu1bit.int8_mma as pkg
    spec = importlib.util.spec_from_file_location(full, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    setattr(pkg, name, mod)
    spec.loader.exec_module(mod)
    return mod


class _Finder:
    """Resolve `import gpu1bit.int8_mma.<wip module>` to this directory."""
    def find_spec(self, fullname, path=None, target=None):
        prefix = "gpu1bit.int8_mma."
        if fullname.startswith(prefix) and fullname[len(prefix):] in MODULES:
            return importlib.util.spec_from_file_location(fullname, HERE / f"{fullname[len(prefix):]}.py")
        return None


sys.meta_path.insert(0, _Finder())
