"""
compare_labels.py

Quick visual spot-check before training on the new SAM2 labels: shows
frame | old algorithmic label (dataset_prep.py) | new SAM2 label,
side by side, for a handful of frames across a few samples. Two minutes
of looking at this now is much cheaper than finding out after a 40-epoch
training run that something's off.

Usage:
    python compare_labels.py --training_data D:\\Minor_Project\\training_data ^
                              --training_data_sam2 D:\\Minor_Project\\training_data_sam2 ^
                              --samples sample_001 sample_005 sample_018 --n_per_sample 3 ^
                              --out compare.png
"""

import argparse
import glob
import os
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training_data", required=True, help="original training_data (has the algorithmic labels)")
    ap.add_argument("--training_data_sam2", required=True, help="new training_data_sam2 (has the SAM2 labels)")
    ap.add_argument("--samples", nargs="+", required=True, help="sample names to spot-check, e.g. sample_001 sample_018")
    ap.add_argument("--n_per_sample", type=int, default=2, help="frames to show per sample (only non-empty ones)")
    ap.add_argument("--out", default="compare.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []  # (frame_path, old_label_path, sam2_label_path)

    for sample_name in args.samples:
        frame_paths = sorted(glob.glob(os.path.join(args.training_data, "frames", f"{sample_name}_*.png")))
        non_empty = []
        for fp in frame_paths:
            name = os.path.basename(fp)
            sam2_label = os.path.join(args.training_data_sam2, "labels", name)
            if os.path.exists(sam2_label):
                m = cv2.imread(sam2_label, cv2.IMREAD_GRAYSCALE)
                if m is not None and m.max() > 0:
                    non_empty.append(fp)
        if not non_empty:
            print(f"[{sample_name}] no non-empty SAM2 labels found -- skipping")
            continue
        picked = rng.sample(non_empty, min(args.n_per_sample, len(non_empty)))
        for fp in picked:
            name = os.path.basename(fp)
            old_label = os.path.join(args.training_data, "labels", name)
            sam2_label = os.path.join(args.training_data_sam2, "labels", name)
            rows.append((fp, old_label, sam2_label))

    if not rows:
        print("Nothing to show -- no non-empty SAM2 labels found for the given samples.")
        return

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for row, (fp, old_p, sam2_p) in enumerate(rows):
        frame = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
        old_m = cv2.imread(old_p, cv2.IMREAD_GRAYSCALE) if os.path.exists(old_p) else None
        sam2_m = cv2.imread(sam2_p, cv2.IMREAD_GRAYSCALE)

        axes[row, 0].imshow(frame)
        axes[row, 0].set_title("frame" if row == 0 else "")
        axes[row, 0].set_ylabel(os.path.basename(fp), fontsize=7)
        if old_m is not None:
            axes[row, 1].imshow(old_m, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title("old (algorithmic)" if row == 0 else "")
        axes[row, 2].imshow(sam2_m, cmap="gray", vmin=0, vmax=255)
        axes[row, 2].set_title("new (SAM2)" if row == 0 else "")
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        for c in (1, 2):
            axes[row, c].axis("off")

    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"Saved {args.out} ({n} frames across {len(args.samples)} sample(s))")


if __name__ == "__main__":
    main()
