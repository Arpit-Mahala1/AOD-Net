"""
mamba_unet_prepare_data.py

Converts training_data/ into a plain image/mask folder pair for a
Mamba U-Net baseline -- using the EXACT same video-level train/val split
train.py uses for the CNN (same seed, same logic, imported directly from
train.py), so this comparison is trained/evaluated on the same split as
run4_seeded, YOLO26, DeepLabV3+, and Mask R-CNN.

Identical logic to dl3_prepare_data.py: dense per-pixel labels, no polygon
conversion needed. Kept as its own script (rather than reusing dl3's) so
each model in the comparison has its own self-contained, independently
runnable pipeline, matching the pattern already established for YOLO26 /
DeepLabV3+ / Mask R-CNN.

NOTE ON SCOPE (same as every other baseline in this comparison):
(a) gets the raw current frame only (3-channel RGB), not the CNN's
    6-channel frame+background input -- no "what changed" signal.
(b) frames are used at 256x256 (training_data's own resolution).

Usage:
    python mamba_unet_prepare_data.py --data_root D:\\Minor_Project\\training_data ^
                                       --out D:\\Minor_Project\\mamba_unet_dataset
"""

import argparse
import glob
import os
import shutil

import cv2

from train import group_split_indices  # reuse the EXACT same split logic + seed as the CNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="training_data folder (frames/ + labels/), same one used to train the CNN")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_fraction", type=float, default=0.2,
                     help="MUST match the --val_fraction used for the CNN model you're comparing against")
    ap.add_argument("--split_seed", type=int, default=42,
                     help="MUST match the --split_seed used for the CNN model you're comparing against")
    args = ap.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.data_root, "frames", "*.png")))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found under {args.data_root}/frames")

    _, _, train_samples, val_samples = group_split_indices(frame_paths, args.val_fraction, args.split_seed)
    train_set, val_set = set(train_samples), set(val_samples)
    print(f"Train videos ({len(train_samples)}): {train_samples}")
    print(f"Val videos   ({len(val_samples)}): {val_samples}")

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "masks", split), exist_ok=True)

    n_obj = n_empty = n_skipped = 0
    for fpath in frame_paths:
        name = os.path.splitext(os.path.basename(fpath))[0]
        sample_name = name.rsplit("_", 1)[0]
        if sample_name in train_set:
            split = "train"
        elif sample_name in val_set:
            split = "val"
        else:
            n_skipped += 1
            continue

        label_path = os.path.join(args.data_root, "labels", name + ".png")
        shutil.copy(fpath, os.path.join(args.out, "images", split, name + ".png"))
        m = None
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(args.out, "masks", split, name + ".png"))
            m = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if m is not None and m.max() > 0:
            n_obj += 1
        else:
            n_empty += 1

    print(f"\nDone. {n_obj} frames with object(s), {n_empty} empty frames"
          f"{f', {n_skipped} skipped (not in either split)' if n_skipped else ''}.")
    print(f"Dataset written to {args.out}")


if __name__ == "__main__":
    main()
