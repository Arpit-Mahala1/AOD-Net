"""
vadmamba_evaluate.py

Evaluates a trained vadmamba_train.py predictor across real val-video
sequences. For each frame t (t >= k), predicts frame t from the k frames
before it, computes a per-pixel prediction-error map, thresholds it into a
binary "anomaly mask", and scores it against the real object mask using
the SAME pixel-level TP/FP/FN accounting as every other baseline in this
comparison -- computed once over ALL val pixels combined.

Unlike every other evaluate script here, there's a free hyperparameter:
--error_percentile controls how much of each frame's error map gets
flagged as "anomalous". This is inherent to the anomaly-detection paradigm
(there's no learned decision boundary the way there is for a classifier) --
worth sweeping a few values and reporting the best, rather than picking
one number and hoping.

See vadmamba_prepare_data.py's docstring for what "VADMamba-inspired"
means here and why.

Usage:
    python vadmamba_evaluate.py --vadmamba_dataset D:\\Minor_Project\\vadmamba_dataset ^
                                 --model D:\\Minor_Project\\runs\\vadmamba\\model.pt ^
                                 --error_percentile 97
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch

from vadmamba_train import MambaPredictor, normalize, denormalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vadmamba_dataset", required=True, help="folder produced by vadmamba_prepare_data.py")
    ap.add_argument("--model", required=True, help="path to model.pt saved by vadmamba_train.py")
    ap.add_argument("--k", type=int, default=4, help="MUST match the value used for training")
    ap.add_argument("--n_mamba_layers", type=int, default=4, help="MUST match the value used for training")
    ap.add_argument("--error_percentile", type=float, default=97.0,
                     help="per-frame pixel-error percentile above which a pixel is flagged anomalous. "
                          "Higher = stricter (fewer false positives, may miss small objects). Try a few values.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MambaPredictor(k=args.k, n_mamba_layers=args.n_mamba_layers).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    images_dir = os.path.join(args.vadmamba_dataset, "images", "val")
    masks_dir = os.path.join(args.vadmamba_dataset, "masks", "val")
    manifest_path = os.path.join(args.vadmamba_dataset, "manifest_val.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    n_scored_frames = sum(max(0, len(frames) - args.k) for frames in manifest.values())
    print(f"Evaluating on {n_scored_frames} val frames (need {args.k} frames of context per prediction, "
          f"so the first {args.k} frames of each video aren't scored)...")

    tp = fp = fn = 0
    correct_frames = 0
    n_done = 0

    with torch.no_grad():
        for video, frames in manifest.items():
            names = [e["frame"] for e in frames]
            for t in range(args.k, len(names)):
                ctx_names = names[t - args.k:t]
                target_name = names[t]

                ctx = [normalize(cv2.imread(os.path.join(images_dir, n))) for n in ctx_names]
                x = torch.from_numpy(np.concatenate(ctx, axis=0)).unsqueeze(0).to(device).float()
                pred = model(x)[0].cpu().numpy()  # C,H,W, normalized space
                pred_img = denormalize(pred)  # H,W,C in [0,1]

                actual_bgr = cv2.imread(os.path.join(images_dir, target_name))
                actual_img = denormalize(normalize(actual_bgr))  # round-trip through same normalization for a fair comparison

                error = np.abs(pred_img - actual_img).mean(axis=2)  # H,W, mean abs error across channels
                thresh = np.percentile(error, args.error_percentile)
                pred_mask = error > thresh

                mask_path = os.path.join(masks_dir, target_name)
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else np.zeros(error.shape, np.uint8)
                gt_pos = gt_mask > 0

                tp += int(np.logical_and(gt_pos, pred_mask).sum())
                fp += int(np.logical_and(~gt_pos, pred_mask).sum())
                fn += int(np.logical_and(gt_pos, ~pred_mask).sum())
                if gt_pos.any() == pred_mask.any():
                    correct_frames += 1

                n_done += 1
                if n_done % 200 == 0:
                    print(f"  ...{n_done}/{n_scored_frames}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    frame_acc = correct_frames / n_done if n_done else float("nan")

    print(f"\nVADMamba-inspired frame-prediction results ({n_done} val frames, "
          f"error_percentile={args.error_percentile}):")
    print(f"  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}  iou={iou:.3f}")
    print(f"  frame-level detection accuracy={frame_acc:.3f}")
    print(f"\nDirectly comparable CNN numbers (run4_seeded, from train.py's own val log):")
    print(f"  precision=0.860  recall=0.453  f1=0.594  iou=0.422")
    print(f"\nNote: --error_percentile is a free knob with no learned equivalent in the other "
          f"baselines -- worth reporting results at 2-3 values (e.g. 90, 95, 97, 99) rather than one, "
          f"since this method's whole operating point depends on it.")
    print(f"\nReminder: this is a frame-level comparison only, not the full P/C-gated video-level "
          f"pipeline the CNN runs through in infer.py -- by design, per the 'quick comparison' scope. "
          f"Also: this is a VADMamba-INSPIRED reproduction (no optical-flow branch, mambapy instead of "
          f"mamba-ssm) -- see vadmamba_prepare_data.py's docstring for why.")


if __name__ == "__main__":
    main()
