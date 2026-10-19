"""
infer_ablation.py

Inference-only ablations for the camera-ready revision. Uses the SAME trained
checkpoint and the same infer.run_on_video() code path as run_pipeline.py --
nothing is retrained. Each run writes prediction folders that
revision_eval.py can score, plus a timing CSV (frames per second).

What each switch tests:
  --bg_mode rolling   the normal pipeline (3 s rolling median) -- baseline
  --bg_mode frame     background channel = current frame, i.e. the
                      "what changed" cue is removed. Tests how much the
                      6-channel fusion actually contributes (R1-f, R2-3).
  --bg_mode first     background frozen at the median of the first 3 s of
                      the video (scenes start empty). Tests whether a
                      long-term background helps during the persistence
                      window (R1-d).
  --disable_flow_check   pipeline without optical-flow verification (R1-f)
  --no_gating         P=0 and an effectively infinite C: every tracked blob
                      is reported immediately and forever. Shows what the
                      temporal gate contributes (R1-f).
  --min_area / --prob_thresh   small-object settings (R3). infer.py
                      discards blobs under 100 px at 256x256 (~0.15% of
                      the frame), close to your smallest objects (0.21%).

Usage (normal pipeline, timed):
    python infer_ablation.py --dataset_root dataset --model weights/aod_net_final.pt ^
        --out_root ablations/rolling --prob_thresh 0.5

Usage (no background cue):
    python infer_ablation.py ... --bg_mode frame --out_root ablations/no_bg

Then score each run:
    python revision_eval.py --dataset_root dataset --predictions_root ablations/no_bg ^
        --data_root training_data --out no_bg.csv
"""

import argparse
import csv
import functools
import glob
import os
import time

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

import infer
from dataset_prep import parse_txt, find_video_file, find_txt_file


class FrameAsBackground:
    """Background channel is a copy of the current frame: no change cue at all."""

    def __init__(self, history_frames):
        self.buffer = []
        self.history_frames = history_frames

    def update(self, frame_bgr):
        return frame_bgr


class FrozenFirstBackground(infer.RollingBackground):
    """Rolling median until the buffer first fills (about 3 s), then frozen."""

    def update(self, frame_bgr):
        if len(self.buffer) >= self.history_frames and self.bg is not None:
            return self.bg
        return super().update(frame_bgr)


BG_CLASSES = {"rolling": infer.RollingBackground, "frame": FrameAsBackground, "first": FrozenFirstBackground}


def build_model(architecture, encoder_depth, model_path, device):
    dec = {5: (256, 128, 64, 32, 16), 4: (256, 128, 64, 32), 3: (256, 128, 64)}
    if architecture == "unet":
        model = smp.Unet(encoder_name="resnet18", encoder_weights=None, encoder_depth=encoder_depth,
                         decoder_channels=dec[encoder_depth], in_channels=6, classes=1)
    else:
        model = smp.DeepLabV3Plus(encoder_name="resnet18", encoder_weights=None, encoder_depth=encoder_depth,
                                  in_channels=6, classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, help="folder of sample_XXX folders (P/C read from each txt)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--architecture", choices=["unet", "deeplabv3plus"], default="unet")
    ap.add_argument("--encoder_depth", type=int, default=5, choices=[3, 4, 5])
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--prob_thresh", type=float, default=0.5,
                    help="use the SAME value as the run that produced the paper's Table 2")
    ap.add_argument("--bg_mode", choices=list(BG_CLASSES), default="rolling")
    ap.add_argument("--disable_flow_check", action="store_true")
    ap.add_argument("--no_gating", action="store_true", help="P=0, C=1e9: report every tracked blob immediately")
    ap.add_argument("--min_area", type=int, default=100, help="min blob area in px at img_size (infer.py uses 100)")
    ap.add_argument("--mask_dilate_px", type=int, default=0)
    ap.add_argument("--samples", nargs="*", default=None, help="only these sample names")
    args = ap.parse_args()

    # Swap in the chosen background model and blob filter without touching infer.py.
    infer.RollingBackground = BG_CLASSES[args.bg_mode]
    infer.extract_blobs = functools.partial(infer.extract_blobs, min_area=args.min_area)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.architecture, args.encoder_depth, args.model, device)
    print(f"device={device} bg_mode={args.bg_mode} flow_check={'off' if args.disable_flow_check else 'on'} "
          f"gating={'off' if args.no_gating else 'on'} min_area={args.min_area} prob_thresh={args.prob_thresh}")

    os.makedirs(args.out_root, exist_ok=True)
    timing_rows = []
    for folder in sorted(glob.glob(os.path.join(args.dataset_root, "sample_*"))):
        name = os.path.basename(os.path.normpath(folder))
        if args.samples and name not in args.samples:
            continue
        video_path = find_video_file(folder)
        p, c, _ = parse_txt(find_txt_file(folder, video_path))
        if args.no_gating:
            p, c = 0, 10 ** 9

        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        infer.run_on_video(video_path, model, device, os.path.join(args.out_root, name), p, c,
                           img_size=args.img_size, prob_thresh=args.prob_thresh,
                           use_flow_check=not args.disable_flow_check, mask_dilate_px=args.mask_dilate_px)
        if device == "cuda":
            torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        fps = n_frames / secs if secs > 0 else float("nan")
        timing_rows.append({"sample": name, "frames": n_frames, "seconds": round(secs, 2), "fps": round(fps, 1)})
        print(f"  {name}: {n_frames} frames in {secs:.1f} s -> {fps:.1f} FPS")

    if timing_rows:
        path = os.path.join(args.out_root, "timing.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
            w.writeheader()
            w.writerows(timing_rows)
        total_f = sum(r["frames"] for r in timing_rows)
        total_s = sum(r["seconds"] for r in timing_rows)
        gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        print(f"\nOverall: {total_f / total_s:.1f} FPS end to end on {gpu} "
              f"(includes decoding, background, model, flow, tracking and writing masks)")
        print(f"Timing CSV: {path}")


if __name__ == "__main__":
    main()
