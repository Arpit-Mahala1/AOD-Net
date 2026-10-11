"""
vadmamba_prepare_data.py

Converts training_data/ into the format needed for a VADMamba-INSPIRED
frame-prediction anomaly detector -- using the EXACT same video-level
train/val split train.py uses for the CNN (same seed, same logic, imported
directly from train.py), so this comparison is trained/evaluated on the
same split as every other model in this comparison.

READ THIS FIRST -- what "VADMamba-inspired" means here:
The official VADMamba (ICME 2025) stacks two Windows-hostile custom-CUDA
dependencies: VM-UNet's mamba-ssm/causal-conv1d, and FlowNet2 for optical
flow (an old, unmaintained repo with its own CUDA kernels, hard to build
even on Linux). Running the literal paper code on your machine risked
burning your whole time budget on installs, not results. This is a
faithful-in-spirit reproduction instead: same core idea (train a predictor
on normal-only frames, flag high prediction-error regions as anomalies),
built with tools already proven to work cleanly here -- the mambapy Mamba
block from your mamba_unet baseline, and OpenCV's Farneback optical flow
(already installed, zero new dependency) instead of FlowNet2. State this
plainly in your write-up.

WHAT COUNTS AS "NORMAL": a frame whose labels/*.png mask is entirely
black (no object present yet). This is derived directly from your existing
label masks -- no new annotation or metadata needed.

Unlike every other baseline in this comparison (which uses independent,
shuffled single frames), this one needs TEMPORAL ORDER, since it predicts
frame t from frames t-k..t-1. So alongside the usual images/+masks/ copy,
this script also writes a manifest.json per split recording, per video,
the time-ordered list of frame filenames and which ones are "normal".

Usage:
    python vadmamba_prepare_data.py --data_root D:\\Minor_Project\\training_data ^
                                     --out D:\\Minor_Project\\vadmamba_dataset
"""

import argparse
import glob
import json
import os
import shutil
from collections import defaultdict

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

    manifests = {"train": defaultdict(list), "val": defaultdict(list)}
    n_normal = n_anomalous = n_skipped = 0

    for fpath in sorted(frame_paths):  # sorted -> filename order -> temporal order within a video
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
        m = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(label_path) else None
        is_normal = m is None or m.max() == 0
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(args.out, "masks", split, name + ".png"))

        manifests[split][sample_name].append({"frame": name + ".png", "normal": bool(is_normal)})
        if is_normal:
            n_normal += 1
        else:
            n_anomalous += 1

    for split in ("train", "val"):
        manifest_path = os.path.join(args.out, f"manifest_{split}.json")
        with open(manifest_path, "w") as f:
            json.dump(manifests[split], f, indent=2)
        n_videos = len(manifests[split])
        n_frames = sum(len(v) for v in manifests[split].values())
        print(f"{split}: {n_videos} videos, {n_frames} frames -> {manifest_path}")

    print(f"\nDone. {n_normal} normal frames, {n_anomalous} anomalous (object-present) frames"
          f"{f', {n_skipped} skipped (not in either split)' if n_skipped else ''}.")
    print(f"Dataset written to {args.out}")


if __name__ == "__main__":
    main()
