"""Run the proven recovery worker with deterministic acoustic augmentation."""
import argparse
from pathlib import Path
from augmentation_features import AugmentedFeatures
from recovery_train import run_worker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("job")
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_worker(args.run, args.job, args.target, args.seconds, args.resume, feature_factory=AugmentedFeatures)
