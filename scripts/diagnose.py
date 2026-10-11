"""
diagnose.py

Quick post-hoc diagnostic against an ALREADY-TRAINED checkpoint -- no
retraining needed. Loads a few known (frame, background, label) triples
directly from training_data/ and shows the RAW probability heatmap next to
the thresholded prediction.

Why: "all blank" at threshold 0.5 has two very different possible causes:
  1. The model has genuinely learned nothing there (max probability near the
     object is close to 0 everywhere) -- a real detection failure, likely
     needs a resolution/architecture/loss-weighting fix.
  2. The model has a weak but real signal (e.g. max probability 0.2-0.45)
     that the 0.5 cutoff is simply throwing away -- a much cheaper fix
     (lower the threshold, or weight the positive class more).

The printed min/max/mean probability numbers tell you which one you're in
before you spend more GPU time guessing.

Usage:
    python diagnose.py --data_root D:\\Minor_Project\\training_data \\
                        --model D:\\Minor_Project\\runs\\run4\\model.pt \\
                        --frames sample_010_001371.png sample_026_002331.png
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import segmentation_models_pytorch as smp

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="the training_data folder (has frames/backgrounds/labels)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", nargs="+", required=True,
                     help="filenames exactly as they appear in training_data/frames/, e.g. sample_010_001371.png")
    ap.add_argument("--out", default="diagnosis.png")
    ap.add_argument("--encoder_depth", type=int, default=5, choices=[3, 4, 5],
                     help="MUST match the value used when this checkpoint was trained")
    ap.add_argument("--architecture", choices=["unet", "deeplabv3plus"], default="unet",
                     help="MUST match what this checkpoint was trained with (train.py's --architecture)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005],
                     help="sweep of decision thresholds to test for free against this checkpoint, no retraining")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder_channels_by_depth = {5: (256, 128, 64, 32, 16), 4: (256, 128, 64, 32), 3: (256, 128, 64)}
    if args.architecture == "unet":
        model = smp.Unet(encoder_name="resnet18", encoder_weights=None, encoder_depth=args.encoder_depth,
                          decoder_channels=decoder_channels_by_depth[args.encoder_depth], in_channels=6, classes=1)
    else:  # deeplabv3plus
        model = smp.DeepLabV3Plus(encoder_name="resnet18", encoder_weights=None, encoder_depth=args.encoder_depth,
                                   in_channels=6, classes=1)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.to(device).eval()

    n = len(args.frames)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    if n == 1:
        axes = axes[None, :]

    for row, fname in enumerate(args.frames):
        frame = cv2.imread(os.path.join(args.data_root, "frames", fname))
        bg = cv2.imread(os.path.join(args.data_root, "backgrounds", fname))
        label = cv2.imread(os.path.join(args.data_root, "labels", fname), cv2.IMREAD_GRAYSCALE)
        if frame is None or bg is None or label is None:
            print(f"[skip] {fname}: could not load frame/background/label from {args.data_root}")
            continue

        x = np.concatenate([normalize(frame), normalize(bg)], axis=0)[None, ...]
        x_t = torch.from_numpy(x).to(device)
        with torch.no_grad():
            logits = model(x_t)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        gt_object_px = int((label > 0).sum())
        # probability specifically where the true object is, vs everywhere else
        prob_at_object = prob[label > 0] if gt_object_px > 0 else np.array([0.0])
        print(f"{fname}: gt_object_px={gt_object_px}  "
              f"prob_at_object(min/mean/max)={prob_at_object.min():.4f}/{prob_at_object.mean():.4f}/{prob_at_object.max():.4f}  "
              f"prob_overall_max={prob.max():.4f}")

        if gt_object_px > 0:
            print(f"    threshold sweep (this checkpoint, no retraining):")
            for t in sorted(args.thresholds, reverse=True):
                pred_pos = prob > t
                tp = int(np.logical_and(pred_pos, label > 0).sum())
                pred_pos_count = int(pred_pos.sum())
                recall_t = tp / gt_object_px
                precision_t = tp / pred_pos_count if pred_pos_count > 0 else float("nan")
                print(f"      t={t:.3f}  recall={recall_t:.3f}  precision={precision_t:.3f}  "
                      f"predicted_positive_px={pred_pos_count}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axes[row, 0].imshow(frame_rgb)
        axes[row, 0].set_title("frame" if row == 0 else "")
        axes[row, 0].set_ylabel(fname, fontsize=7)
        axes[row, 1].imshow(label, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title("ground truth" if row == 0 else "")
        im = axes[row, 2].imshow(prob, cmap="jet", vmin=0, vmax=1)
        axes[row, 2].set_title("raw probability (0-1)" if row == 0 else "")
        pred05 = (prob > 0.5).astype(np.uint8) * 255
        axes[row, 3].imshow(pred05, cmap="gray", vmin=0, vmax=255)
        axes[row, 3].set_title("prediction @0.5" if row == 0 else "")
        for c in (1, 2, 3):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

    fig.colorbar(im, ax=axes[:, 2], fraction=0.02, pad=0.02, label="probability")
    fig.savefig(args.out, dpi=110)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
