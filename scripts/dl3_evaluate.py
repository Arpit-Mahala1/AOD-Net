"""
dl3_evaluate.py

Evaluates a trained dl3_train.py checkpoint on the val split produced by
dl3_prepare_data.py, using the SAME pixel-level TP/FP/FN accounting
train.py uses for the CNN's own validation metrics -- computed once at the
end over ALL val pixels combined (not averaged per-image), so the numbers
are directly comparable across every model in this comparison, including
the CNN and YOLO26:
    tp = (gt & pred).sum(); fp = (~gt & pred).sum(); fn = (gt & ~pred).sum()
    precision = tp/(tp+fp); recall = tp/(tp+fn)
    f1 = 2PR/(P+R); iou = tp/(tp+fp+fn)

NOTE ON SCOPE (same limitations as yolo_evaluate.py, restated here since
this is a standalone script): (a) DeepLabV3+ only sees the raw current
frame, not the CNN's 6-channel frame+background input -- no "what changed"
signal; (b) frames are evaluated at 256x256 to match training_data, not
this backbone's more typical higher-resolution usage.

Usage:
    python dl3_evaluate.py --dl3_dataset D:\\Minor_Project\\dl3_dataset ^
                            --model D:\\Minor_Project\\runs\\dl3\\model.pt
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)


def compute_pixel_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    return precision, recall, f1, iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl3_dataset", required=True, help="folder produced by dl3_prepare_data.py")
    ap.add_argument("--model", required=True, help="path to model.pt saved by dl3_train.py")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # weights_backbone=None: we're loading a fully-trained checkpoint next, so
    # there's no reason to download ImageNet backbone weights just to overwrite them
    model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=2)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device).eval()

    images_dir = os.path.join(args.dl3_dataset, "images", "val")
    masks_dir = os.path.join(args.dl3_dataset, "masks", "val")
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
            pred = model(x)["out"].argmax(dim=1)[0].cpu().numpy()

            gt_pos = gt_mask > 0
            pred_pos = pred > 0
            tp += int(np.logical_and(gt_pos, pred_pos).sum())
            fp += int(np.logical_and(~gt_pos, pred_pos).sum())
            fn += int(np.logical_and(gt_pos, ~pred_pos).sum())
            if gt_pos.any() == pred_pos.any():
                correct_frames += 1

            if (i + 1) % 200 == 0:
                print(f"  ...{i + 1}/{len(val_images)}")

    precision, recall, f1, iou = compute_pixel_metrics(tp, fp, fn)
    frame_acc = correct_frames / len(val_images)

    print(f"\nDeepLabV3+ frame-level results ({len(val_images)} val frames):")
    print(f"  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}  iou={iou:.3f}")
    print(f"  frame-level detection accuracy={frame_acc:.3f}")
    print(f"\nDirectly comparable CNN numbers (run4_seeded, from train.py's own val log):")
    print(f"  precision=0.860  recall=0.453  f1=0.594  iou=0.422")
    print(f"\nReminder: this is a frame-level comparison only (does DeepLabV3+ find the object in a "
          f"single frame), not the full P/C-gated video-level pipeline the CNN runs through in "
          f"infer.py -- by design, per the 'quick comparison' scope.")


if __name__ == "__main__":
    main()
