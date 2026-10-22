"""Run a WIP module's unittest suite under one B gpu lock:  shared/python run.py test_parts [Class.test]."""
import sys
sys.path.insert(0, "/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/B/wip/parts")
import wip  # noqa: F401,E402
from gpu1bit.int8_mma import gpulock  # noqa: E402
names = ["gpu1bit.int8_mma." + (sys.argv[1] if len(sys.argv) > 1 else "test_parts")]
if len(sys.argv) > 2:
    names = [names[0] + "." + a for a in sys.argv[2:]]
gpulock.run_unittests("B:x1/x2 tests (wip)", names)
