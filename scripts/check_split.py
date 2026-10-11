"""
check_split.py

Reproduces train.py's group_split_indices() WITHOUT training anything --
just tells you which videos landed in train vs val for a given data_root
and seed. Use this to check whether a specific problem sample (e.g. the one
diagnose.py flagged as near-zero probability) was ever seen during training.

Usage:
    python check_split.py --data_root D:\\Minor_Project\\training_data_sam2 ^
                           --check sample_010 sample_026
"""

import argparse
import glob
import os
import random
from collections import defaultdict


def group_split_indices(frame_paths, val_fraction=0.2, seed=42):
    groups = defaultdict(list)
    for idx, fpath in enumerate(frame_paths):
        name = os.path.splitext(os.path.basename(fpath))[0]
        sample_name = name.rsplit("_", 1)[0]
        groups[sample_name].append(idx)

    sample_names = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(sample_names)

    n_val = max(1, round(len(sample_names) * val_fraction))
    val_samples = sorted(sample_names[:n_val])
    train_samples = sorted(sample_names[n_val:])
    return train_samples, val_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--val_fraction", type=float, default=0.2, help="MUST match what train.py used")
    ap.add_argument("--split_seed", type=int, default=42, help="MUST match what train.py used")
    ap.add_argument("--check", nargs="+", default=[], help="sample names to look up, e.g. sample_010 sample_026")
    args = ap.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.data_root, "frames", "*.png")))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found under {args.data_root}/frames -- check --data_root")

    train_samples, val_samples = group_split_indices(frame_paths, args.val_fraction, args.split_seed)
    print(f"Train videos ({len(train_samples)}): {train_samples}")
    print(f"Val videos   ({len(val_samples)}): {val_samples}")

    if args.check:
        print()
        for s in args.check:
            if s in val_samples:
                print(f"{s}: VAL (held out -- model never trained on this video)")
            elif s in train_samples:
                print(f"{s}: TRAIN (model trained directly on this video)")
            else:
                print(f"{s}: not found in either split -- check the name/data_root")


if __name__ == "__main__":
    main()
