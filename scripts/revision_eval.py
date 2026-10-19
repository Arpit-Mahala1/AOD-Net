"""
revision_eval.py

Extra metrics for the camera-ready revision, computed from prediction folders
that infer.py / run_pipeline.py / infer_ablation.py have ALREADY written.
No model and no GPU needed -- this only reads PNGs.

Per video it reports the same pixel metrics as evaluate_predictions.py, plus:
  alarm_recall        of the seconds where the ground truth contains an object,
                      the fraction where the prediction is non-empty
  alarm_specificity   of the seconds where the ground truth is empty,
                      the fraction where the prediction is also empty
  balanced_accuracy   mean of the two (not inflated by long empty stretches)
  event_detected      1 if any predicted pixel overlaps the ground truth in any
                      positive second, else 0
  second_level_detection_accuracy   the existing metric, unchanged

Summaries are printed for all videos and, when --data_root (the
training_data folder) or --val is given, separately for the train and
held-out videos, using the same split as train.py.

Usage (custom dataset, split taken from training_data):
    python revision_eval.py --dataset_root dataset ^
        --predictions_root pipeline_out/predictions ^
        --data_root training_data --out revision_eval.csv

Usage (ABODA, no split):
    python revision_eval.py --dataset_root aboda --predictions_root aboda_out/predictions ^
        --out aboda_revision_eval.csv

Usage (held-out videos named explicitly):
    python revision_eval.py ... --val sample_003 sample_011 sample_017 sample_020 sample_024
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np

from dataset_prep import parse_txt, find_video_file, find_txt_file
from evaluate_predictions import build_gt_by_second, mask_number_to_second
from check_split import group_split_indices


def safe_div(a, b):
    return a / b if b > 0 else float("nan")


def evaluate_sample(sample_folder, masks_dir):
    video_path = find_video_file(sample_folder)
    txt_path = find_txt_file(sample_folder, video_path)
    p, c, intervals = parse_txt(txt_path)

    pred_paths = sorted(glob.glob(os.path.join(masks_dir, "mask*.png")), key=mask_number_to_second)
    if not pred_paths:
        return None

    target_shape = cv2.imread(pred_paths[0], cv2.IMREAD_GRAYSCALE).shape
    gt_by_second = build_gt_by_second(sample_folder, intervals, target_shape)
    pred_by_second = {mask_number_to_second(pp): pp for pp in pred_paths}
    all_seconds = sorted(set(gt_by_second.keys()) | set(pred_by_second.keys()))
    zeros = np.zeros(target_shape, dtype=np.uint8)

    tp = fp = fn = 0
    pos_secs = pos_hit = neg_secs = neg_ok = correct = 0
    overlap_any = False
    for sec in all_seconds:
        gt = gt_by_second.get(sec, zeros)
        pred = zeros
        if sec in pred_by_second:
            pred = cv2.imread(pred_by_second[sec], cv2.IMREAD_GRAYSCALE)
            if pred is None:
                pred = zeros
            elif pred.shape != target_shape:
                pred = cv2.resize(pred, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        g, q = gt > 0, pred > 0
        sec_tp = int(np.logical_and(g, q).sum())
        tp += sec_tp
        fp += int(np.logical_and(~g, q).sum())
        fn += int(np.logical_and(g, ~q).sum())

        g_any, q_any = bool(g.any()), bool(q.any())
        correct += int(g_any == q_any)
        if g_any:
            pos_secs += 1
            pos_hit += int(q_any)
            overlap_any = overlap_any or sec_tp > 0
        else:
            neg_secs += 1
            neg_ok += int(not q_any)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if precision == precision and recall == recall else float("nan")
    alarm_recall = safe_div(pos_hit, pos_secs)
    alarm_spec = safe_div(neg_ok, neg_secs)
    bal = np.nanmean([alarm_recall, alarm_spec]) if (pos_secs or neg_secs) else float("nan")

    return {
        "sample": os.path.basename(os.path.normpath(sample_folder)),
        "p": p, "c": c,
        "seconds_evaluated": len(all_seconds),
        "positive_seconds": pos_secs, "negative_seconds": neg_secs,
        "predicted_pixels": tp + fp,
        "pixel_precision": precision, "pixel_recall": recall, "pixel_f1": f1,
        "pixel_iou": safe_div(tp, tp + fp + fn),
        "second_level_detection_accuracy": safe_div(correct, len(all_seconds)),
        "alarm_recall": alarm_recall,
        "alarm_specificity": alarm_spec,
        "balanced_accuracy": bal,
        "event_detected": int(overlap_any),
    }


SUMMARY_KEYS = ["pixel_precision", "pixel_recall", "pixel_f1", "pixel_iou",
                "second_level_detection_accuracy", "alarm_recall", "alarm_specificity",
                "balanced_accuracy", "event_detected"]


def summarize(rows, label):
    if not rows:
        return
    print(f"\n{label}: {len(rows)} video(s)")
    for k in SUMMARY_KEYS:
        vals = [r[k] for r in rows]
        # NaN counted as 0, the same convention the paper uses for Tables 2 and 4
        mean = sum(v if v == v else 0.0 for v in vals) / len(vals)
        print(f"  {k:34s} {mean:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, help="folder with sample_XXX ground-truth folders")
    ap.add_argument("--predictions_root", required=True, help="folder with one prediction subfolder per sample")
    ap.add_argument("--data_root", default=None,
                    help="training_data folder used by train.py; used only to recover the train/val split")
    ap.add_argument("--val_fraction", type=float, default=0.2, help="must match train.py")
    ap.add_argument("--split_seed", type=int, default=42, help="must match train.py")
    ap.add_argument("--val", nargs="*", default=None, help="held-out sample names, if you'd rather list them")
    ap.add_argument("--out", default="revision_eval.csv")
    args = ap.parse_args()

    val_set = None
    if args.val:
        val_set = set(args.val)
    elif args.data_root:
        frame_paths = sorted(glob.glob(os.path.join(args.data_root, "frames", "*.png")))
        if not frame_paths:
            raise FileNotFoundError(f"No frames under {args.data_root}/frames")
        _, val_samples = group_split_indices(frame_paths, args.val_fraction, args.split_seed)
        val_set = set(val_samples)

    rows = []
    for folder in sorted(glob.glob(os.path.join(args.dataset_root, "sample_*"))):
        name = os.path.basename(os.path.normpath(folder))
        masks_dir = os.path.join(args.predictions_root, name)
        if not os.path.isdir(masks_dir):
            print(f"[skip] {name}: no predictions at {masks_dir}")
            continue
        r = evaluate_sample(folder, masks_dir)
        if r is None:
            print(f"[skip] {name}: prediction folder is empty")
            continue
        r["split"] = ("val" if name in val_set else "train") if val_set is not None else "-"
        rows.append(r)
        print(f"[ok] {name} ({r['split']}): IoU={r['pixel_iou']:.3f}  det_acc={r['second_level_detection_accuracy']:.3f}  "
              f"alarm_recall={r['alarm_recall']:.3f}  alarm_spec={r['alarm_specificity']:.3f}  "
              f"pred_pixels={r['predicted_pixels']}")

    if not rows:
        print("Nothing evaluated.")
        return

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summarize(rows, "ALL videos")
    if val_set is not None:
        summarize([r for r in rows if r["split"] == "train"], "TRAIN videos (seen during training)")
        summarize([r for r in rows if r["split"] == "val"], "HELD-OUT videos (never seen during training)")
    print(f"\nPer-video report: {args.out}")


if __name__ == "__main__":
    main()
