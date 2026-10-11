"""
evaluate_predictions.py

Compares infer.py's predicted masks against the ORIGINAL ground-truth mask
intervals for a sample_XXX-style video (the same 26-video dataset format:
video_XX.txt with P/C + "maskN.png: STARTs-ENDs" lines, plus the maskN.png
files themselves). This is what lets you say "how well does this model do
on a dataset with known answers" -- not just eyeball it with
review_output.py.

Ground truth here is the RAW given intervals (already P/C-gated by whoever
labeled the dataset) -- NOT the ungated per-object labels dataset_prep.py
builds for training. Those are two different things for two different jobs:
dataset_prep.py needs an ungated label so the CNN has something learnable
per-frame; evaluation needs the actual gated ground truth, since infer.py's
OUTPUT is already gated and that's what should be judged against.

IMPORTANT: infer.py names its output files mask{second+1}.png (1-indexed --
mask1.png is second 0, mask2.png is second 1, etc). This script accounts for
that explicitly. Get it wrong and every comparison is silently off by a
second.

Usage (single video):
    python evaluate_predictions.py --sample_folder D:\\Minor_Project\\dataset\\sample_010 ^
                                    --masks_dir D:\\Minor_Project\\outputs_depth4\\sample_010 ^
                                    --out sample_010_eval.csv

Usage (batch -- evaluate every sample_XXX folder that has a matching predictions folder):
    python evaluate_predictions.py --dataset_root D:\\Minor_Project\\dataset ^
                                    --predictions_root D:\\Minor_Project\\outputs_depth4 ^
                                    --out eval_report.csv
"""

import argparse
import csv
import glob
import os
import re

import cv2
import numpy as np

from dataset_prep import parse_txt, find_video_file, find_txt_file


def build_gt_by_second(folder, intervals, target_shape):
    gt_by_second = {}
    for fname, start, end in intervals:
        mpath = os.path.join(folder, fname)
        if not os.path.exists(mpath):
            continue
        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != target_shape:
            m = cv2.resize(m, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        binary = (m > 0).astype(np.uint8) * 255
        for sec in range(start, end + 1):
            if sec not in gt_by_second:
                gt_by_second[sec] = binary.copy()
            else:
                gt_by_second[sec] = cv2.bitwise_or(gt_by_second[sec], binary)
    return gt_by_second


def mask_number_to_second(path):
    m = re.search(r"mask(\d+)\.png", os.path.basename(path))
    return int(m.group(1)) - 1  # infer.py writes mask{second+1}.png


def evaluate_sample(sample_folder, masks_dir):
    video_path = find_video_file(sample_folder)
    txt_path = find_txt_file(sample_folder, video_path)
    p, c, intervals = parse_txt(txt_path)

    pred_paths = sorted(glob.glob(os.path.join(masks_dir, "mask*.png")), key=mask_number_to_second)
    if not pred_paths:
        return None

    sample_pred = cv2.imread(pred_paths[0], cv2.IMREAD_GRAYSCALE)
    target_shape = sample_pred.shape
    gt_by_second = build_gt_by_second(sample_folder, intervals, target_shape)

    pred_by_second = {mask_number_to_second(p): p for p in pred_paths}
    all_seconds = sorted(set(gt_by_second.keys()) | set(pred_by_second.keys()))
    zeros = np.zeros(target_shape, dtype=np.uint8)

    tp = fp = fn = 0
    correct_seconds = 0
    for sec in all_seconds:
        gt = gt_by_second.get(sec, zeros)
        if sec in pred_by_second:
            pred = cv2.imread(pred_by_second[sec], cv2.IMREAD_GRAYSCALE)
            if pred is None:
                pred = zeros
            elif pred.shape != target_shape:
                pred = cv2.resize(pred, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            pred = zeros

        gt_pos = gt > 0
        pred_pos = pred > 0
        tp += int(np.logical_and(gt_pos, pred_pos).sum())
        fp += int(np.logical_and(~gt_pos, pred_pos).sum())
        fn += int(np.logical_and(gt_pos, ~pred_pos).sum())

        if gt_pos.any() == pred_pos.any():
            correct_seconds += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    detection_accuracy = correct_seconds / len(all_seconds) if all_seconds else float("nan")

    return {
        "sample": os.path.basename(os.path.normpath(sample_folder)),
        "p": p, "c": c,
        "seconds_evaluated": len(all_seconds),
        "pixel_precision": precision, "pixel_recall": recall, "pixel_f1": f1, "pixel_iou": iou,
        "second_level_detection_accuracy": detection_accuracy,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_folder", help="single video mode: one sample_XXX folder")
    ap.add_argument("--masks_dir", help="single video mode: infer.py's output folder for that video")
    ap.add_argument("--dataset_root", help="batch mode: folder containing sample_XXX folders (with ground truth)")
    ap.add_argument("--predictions_root", help="batch mode: parent folder with one subfolder per sample, matching infer.py's --out_root layout")
    ap.add_argument("--out", default="eval_report.csv")
    args = ap.parse_args()

    rows = []
    if args.sample_folder:
        if not args.masks_dir:
            raise ValueError("--masks_dir is required with --sample_folder")
        r = evaluate_sample(args.sample_folder, args.masks_dir)
        if r:
            rows.append(r)
        else:
            print(f"No predicted masks found in {args.masks_dir}")
    elif args.dataset_root:
        if not args.predictions_root:
            raise ValueError("--predictions_root is required with --dataset_root")
        folders = sorted(glob.glob(os.path.join(args.dataset_root, "sample_*")))
        for folder in folders:
            sample_name = os.path.basename(os.path.normpath(folder))
            masks_dir = os.path.join(args.predictions_root, sample_name)
            if not os.path.isdir(masks_dir):
                print(f"[skip] {sample_name}: no predictions folder at {masks_dir}")
                continue
            r = evaluate_sample(folder, masks_dir)
            if r:
                rows.append(r)
                print(f"[ok] {sample_name}: IoU={r['pixel_iou']:.3f} F1={r['pixel_f1']:.3f} "
                      f"detection_acc={r['second_level_detection_accuracy']:.3f}")
    else:
        raise ValueError("Provide either --sample_folder/--masks_dir or --dataset_root/--predictions_root")

    if not rows:
        print("Nothing evaluated.")
        return

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    avg_iou = sum(r["pixel_iou"] for r in rows if r["pixel_iou"] == r["pixel_iou"]) / n
    avg_f1 = sum(r["pixel_f1"] for r in rows if r["pixel_f1"] == r["pixel_f1"]) / n
    avg_det = sum(r["second_level_detection_accuracy"] for r in rows) / n
    print(f"\n{n} video(s) evaluated. Mean pixel IoU={avg_iou:.3f}  Mean pixel F1={avg_f1:.3f}  "
          f"Mean second-level detection accuracy={avg_det:.3f}")
    print(f"Full per-video report saved to {args.out}")


if __name__ == "__main__":
    main()
