"""
dl3_prepare_data.py

Converts training_data/ (frames + labels PNGs, already built by
dataset_prep.py) into a plain image/mask folder pair for a
torchvision DeepLabV3+ (ResNet-50, ImageNet+COCO pretrained) baseline --
using the EXACT same video-level train/val split train.py uses for the CNN
(same seed, same logic, imported directly from train.py rather than
reimplemented), so this comparison is trained/evaluated on the same data
split as run4_seeded and YOLO26.

No format conversion is needed beyond copying: torchvision's segmentation
models want a dense per-pixel mask, which is exactly what dataset_prep.py's
labels/*.png already are. This is simpler than yolo_prepare_data.py, which
had to convert masks -> polygons for YOLO's label format.

NOTE ON SCOPE (quick comparison, not full pipeline integration):
- DeepLabV3+ here gets the raw current frame only (3-channel RGB), NOT the
  frame+background 6-channel input the CNN uses. torchvision's DeepLabV3+
  has no built-in way to consume a 6-channel input without replacing the
  first conv layer (architecture surgery), which is out of scope for a
  quick comparison -- same acknowledged gap as YOLO26.
- Frames are used at 256x256 (training_data's own resolution) to keep the
  comparison on identical images to the CNN, even though this backbone is
  more commonly used at higher resolutions on datasets like Cityscapes/COCO.

Usage:
    python dl3_prepare_data.py --data_root D:\\Minor_Project\\training_data ^
                                --out D:\\Minor_Project\\dl3_dataset
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
