"""`python wip_main.py <parts_run args>`: parts_run.main() with the WIP modules loadable (see wip.py)."""
import sys
sys.path.insert(0, "/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/B/wip/parts")
import wip  # noqa: F401,E402
from gpu1bit.int8_mma import parts_run  # noqa: E402
parts_run.main()
