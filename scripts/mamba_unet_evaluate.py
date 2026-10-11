"""
mamba_unet_evaluate.py

Evaluates a trained mamba_unet_train.py checkpoint on the val split
produced by mamba_unet_prepare_data.py, using the SAME pixel-level
TP/FP/FN accounting train.py uses for the CNN's own validation metrics --
computed once at the end over ALL val pixels combined, so the numbers are
directly comparable across every model in this comparison.

Usage:
    python mamba_unet_evaluate.py --mamba_unet_dataset D:\\Minor_Project\\mamba_unet_dataset ^
                                   --model D:\\Minor_Project\\runs\\mamba_unet\\model.pt
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch

from mamba_unet_train import MambaUNet, normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mamba_unet_dataset", required=True, help="folder produced by mamba_unet_prepare_data.py")
    ap.add_argument("--model", required=True, help="path to model.pt saved by mamba_unet_train.py")
    ap.add_argument("--n_mamba_layers", type=int, default=4, help="MUST match the value used for training")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MambaUNet(n_mamba_layers=args.n_mamba_layers).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    images_dir = os.path.join(args.mamba_unet_dataset, "images", "val")
    masks_dir = os.path.join(args.mamba_unet_dataset, "masks", "val")
    val_images = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if not val_images:
        raise FileNotFoundError(f"No val images found under {images_dir}")
    print(f"Evaluating on {len(val_images)} val frames...")

    tp = fp = fn = 0
    correct_frames = 0

    with torch.no_grad():
        for i, img_path in enumerate(val_images):
            name = os.path.basename(img_path)
            frame = cv2.imread(img_path)
            mask_path = os.path.join(masks_dir, name)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else np.zeros(frame.shape[:2], np.uint8)

            x = torch.from_numpy(normalize(frame)).unsqueeze(0).to(device).float()
            pred = model(x).argmax(dim=1)[0].cpu().numpy()

            gt_pos = gt_mask > 0
            pred_pos = pred > 0
            tp += int(np.logical_and(gt_pos, pred_pos).sum())
            fp += int(np.logical_and(~gt_pos, pred_pos).sum())
            fn += int(np.logical_and(gt_pos, ~pred_pos).sum())
            if gt_pos.any() == pred_pos.any():
                correct_frames += 1

            if (i + 1) % 200 == 0:
                print(f"  ...{i + 1}/{len(val_images)}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    frame_acc = correct_frames / len(val_images)

    print(f"\nMamba U-Net frame-level results ({len(val_images)} val frames):")
    print(f"  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}  iou={iou:.3f}")
    print(f"  frame-level detection accuracy={frame_acc:.3f}")
    print(f"\nDirectly comparable CNN numbers (run4_seeded, from train.py's own val log):")
    print(f"  precision=0.860  recall=0.453  f1=0.594  iou=0.422")
    print(f"\nReminder: this is a frame-level comparison only (does Mamba U-Net find the object in a "
          f"single frame), not the full P/C-gated video-level pipeline the CNN runs through in "
          f"infer.py -- by design, per the 'quick comparison' scope.")


if __name__ == "__main__":
    main()
