"""
yolo_prepare_data.py

Converts training_data/ (frames + labels PNGs, already built by
dataset_prep.py) into a YOLO segmentation dataset -- using the EXACT same
video-level train/val split train.py uses for the CNN (same seed, same
logic, imported directly from train.py rather than reimplemented), so the
YOLO26 comparison is trained and evaluated on the same data split as
run4_seeded. That's what makes this a fair comparison, not just a fast one.

Ground truth polygons are extracted directly from the existing label PNGs
via contour extraction -- no re-labeling needed.

NOTE ON SCOPE (quick comparison, not full pipeline integration):
- YOLO gets the raw current frame only (3-channel RGB), NOT the
  frame+background 6-channel input the CNN uses. YOLO doesn't have a
  natural way to consume a 6-channel input without custom architecture
  surgery, which is out of scope for a quick comparison. This is an
  acknowledged difference worth stating plainly in any write-up: the CNN
  gets a "what changed vs background" signal for free, YOLO doesn't.
- Frames are used at whatever resolution training_data was built at
  (256x256 by default) to keep the comparison on identical images to the
  CNN, even though YOLO26 is more commonly used at imgsz=640. Also worth
  stating as a limitation, not hidden.

Usage:
    python yolo_prepare_data.py --data_root D:\\Minor_Project\\training_data ^
                                 --out D:\\Minor_Project\\yolo_dataset
"""

import argparse
import glob
import os
import shutil

import cv2
import numpy as np

from train import group_split_indices  # reuse the EXACT same split logic + seed as the CNN


def mask_to_yolo_lines(mask_path, min_area=20):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return []
    h, w = m.shape
    binary = (m > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        pts = cnt.reshape(-1, 2).astype(np.float32)
        if len(pts) < 3:
            continue
        pts[:, 0] /= w
        pts[:, 1] /= h
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
        lines.append(f"0 {coords}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="training_data folder (frames/ + labels/), same one used to train the CNN")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_fraction", type=float, default=0.2,
                     help="MUST match the --val_fraction used for the CNN model you're comparing against")
    ap.add_argument("--split_seed", type=int, default=42,
                     help="MUST match the --split_seed used for the CNN model you're comparing against")
    ap.add_argument("--min_area", type=int, default=20, help="min contour area (px, at training_data's own resolution) to keep as a polygon")
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
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

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
        lines = mask_to_yolo_lines(label_path, args.min_area) if os.path.exists(label_path) else []
        if lines:
            n_obj += 1
        else:
            n_empty += 1

        shutil.copy(fpath, os.path.join(args.out, "images", split, name + ".png"))
        with open(os.path.join(args.out, "labels", split, name + ".txt"), "w") as f:
            f.write("\n".join(lines))

    data_yaml_path = os.path.join(args.out, "data.yaml")
    with open(data_yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n  0: object\n")

    print(f"\nDone. {n_obj} frames with object(s), {n_empty} empty frames"
          f"{f', {n_skipped} skipped (not in either split)' if n_skipped else ''}.")
    print(f"Dataset written to {args.out}")
    print(f"data.yaml: {data_yaml_path}")


if __name__ == "__main__":
    main()
