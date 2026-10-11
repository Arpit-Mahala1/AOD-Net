"""
maskrcnn_evaluate.py

Evaluates a trained maskrcnn_train.py checkpoint on the val split produced
by maskrcnn_prepare_data.py, using the SAME pixel-level TP/FP/FN accounting
train.py uses for the CNN's own validation metrics -- computed once at the
end over ALL val pixels combined (not averaged per-image), so the numbers
are directly comparable across every model in this comparison:
    tp = (gt & pred).sum(); fp = (~gt & pred).sum(); fn = (gt & ~pred).sum()
    precision = tp/(tp+fp); recall = tp/(tp+fn)
    f1 = 2PR/(P+R); iou = tp/(tp+fp+fn)

Per-frame prediction masks are the UNION of all predicted instance masks
that clear --conf, matching how yolo_evaluate.py unions multiple YOLO
instance masks into one frame-level mask before scoring -- so an instance
model isn't penalized or favored for splitting one object into two
predictions.

NOTE ON SCOPE (same limitations as yolo_evaluate.py / dl3_evaluate.py):
(a) Mask R-CNN only sees the raw current frame, not the CNN's 6-channel
    frame+background input; (b) frames are evaluated at 256x256 to match
    training_data, not this model's more typical higher-resolution usage.

Usage:
    python maskrcnn_evaluate.py --maskrcnn_dataset D:\\Minor_Project\\maskrcnn_dataset ^
                                 --model D:\\Minor_Project\\runs\\maskrcnn\\model.pt ^
                                 --conf 0.3
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_model():
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes=2)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maskrcnn_dataset", required=True, help="folder produced by maskrcnn_prepare_data.py")
    ap.add_argument("--model", required=True, help="path to model.pt saved by maskrcnn_train.py")
    ap.add_argument("--conf", type=float, default=0.3, help="score threshold for keeping a predicted instance")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # weights_backbone=None: loading a fully-trained checkpoint next, no reason
    # to download COCO/ImageNet weights just to overwrite them
    model = build_model().to(device).eval()
    model.load_state_dict(torch.load(args.model, map_location=device))

    images_dir = os.path.join(args.maskrcnn_dataset, "images", "val")
    masks_dir = os.path.join(args.maskrcnn_dataset, "masks", "val")
    val_images = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if not val_images:
        raise FileNotFoundError(f"No val images found under {images_dir}")
    print(f"Evaluating on {len(val_images)} val frames (conf={args.conf})...")

    tp = fp = fn = 0
    correct_frames = 0

    with torch.no_grad():
        for i, img_path in enumerate(val_images):
            name = os.path.basename(img_path)
            frame = cv2.imread(img_path)
            h, w = frame.shape[:2]
            mask_path = os.path.join(masks_dir, name)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else np.zeros((h, w), np.uint8)

            img_t = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0)
            pred = model([img_t.to(device)])[0]

            pred_mask = np.zeros((h, w), dtype=np.uint8)
            for score, m in zip(pred["scores"].cpu().numpy(), pred["masks"].cpu().numpy()):
                if score >= args.conf:
                    pred_mask = np.maximum(pred_mask, (m[0] > 0.5).astype(np.uint8) * 255)

            gt_pos = gt_mask > 0
            pred_pos = pred_mask > 0
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

    print(f"\nMask R-CNN frame-level results ({len(val_images)} val frames, conf={args.conf}):")
    print(f"  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}  iou={iou:.3f}")
    print(f"  frame-level detection accuracy={frame_acc:.3f}")
    print(f"\nDirectly comparable CNN numbers (run4_seeded, from train.py's own val log):")
    print(f"  precision=0.860  recall=0.453  f1=0.594  iou=0.422")
    print(f"\nReminder: this is a frame-level comparison only (does Mask R-CNN find the object in a "
          f"single frame), not the full P/C-gated video-level pipeline the CNN runs through in "
          f"infer.py -- by design, per the 'quick comparison' scope.")


if __name__ == "__main__":
    main()
