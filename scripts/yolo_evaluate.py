"""
yolo_evaluate.py

Evaluates a trained YOLO26-seg checkpoint on the val split produced by
yolo_prepare_data.py, using the SAME pixel-level TP/FP/FN accounting
train.py uses for the CNN's own validation metrics -- so the resulting
precision/recall/F1/IoU numbers are directly comparable to what train.py
printed for run4_seeded (precision=0.860 recall=0.453 f1=0.594 iou=0.422),
not just eyeballed side by side.

Also reports a simple per-frame "was an object correctly flagged present
or absent" accuracy, comparable in spirit to evaluate_predictions.py's
detection accuracy for the full CNN pipeline (though this is a frame-level
check, not the full P/C-gated video-level one -- this script only covers
the "quick comparison" scope: does YOLO find the object in a frame at all).

Usage:
    python yolo_evaluate.py --yolo_dataset D:\\Minor_Project\\yolo_dataset ^
                             --model D:\\Minor_Project\\runs\\segment\\train\\weights\\best.pt ^
                             --conf 0.25
"""

import argparse
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO


def load_gt_mask(label_path, h, w):
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    if not os.path.exists(label_path):
        return gt_mask
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7:  # class_id + at least 3 points (x,y pairs)
                continue
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            cv2.fillPoly(gt_mask, [coords.astype(np.int32)], 255)
    return gt_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_dataset", required=True, help="folder produced by yolo_prepare_data.py")
    ap.add_argument("--model", required=True, help="path to trained YOLO weights, e.g. runs/segment/train/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--img_size", type=int, default=256)
    args = ap.parse_args()

    model = YOLO(args.model)

    val_images = sorted(glob.glob(os.path.join(args.yolo_dataset, "images", "val", "*.png")))
    if not val_images:
        raise FileNotFoundError(f"No val images found under {args.yolo_dataset}/images/val")
    print(f"Evaluating on {len(val_images)} val frames...")

    tp = fp = fn = 0
    correct_frames = 0

    for i, img_path in enumerate(val_images):
        name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(args.yolo_dataset, "labels", "val", name + ".txt")

        frame = cv2.imread(img_path)
        h, w = frame.shape[:2]
        gt_mask = load_gt_mask(label_path, h, w)

        result = model.predict(img_path, conf=args.conf, imgsz=args.img_size, verbose=False)[0]
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        if result.masks is not None:
            for m in result.masks.data.cpu().numpy():
                m_rs = cv2.resize((m > 0.5).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
                pred_mask = cv2.bitwise_or(pred_mask, m_rs)

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

    print(f"\nYOLO26 frame-level results ({len(val_images)} val frames, conf={args.conf}):")
    print(f"  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}  iou={iou:.3f}")
    print(f"  frame-level detection accuracy={frame_acc:.3f}")
    print(f"\nDirectly comparable CNN numbers (run4_seeded, from train.py's own val log):")
    print(f"  precision=0.860  recall=0.453  f1=0.594  iou=0.422")
    print(f"\nReminder: this is a frame-level comparison only (does YOLO find the object in a "
          f"single frame), not the full P/C-gated video-level pipeline the CNN runs through in "
          f"infer.py -- by design, per the 'quick comparison' scope.")


if __name__ == "__main__":
    main()
