"""
review_output.py

Quick visual review of infer.py's output masks for one test video, without
manually opening every PNG. Prints which seconds have a detection (and the
contiguous ranges, which is what actually matters given the P/C gating), and
saves a montage of frame+mask overlays at the most informative seconds
(start/middle/end of each detection range) so you can eyeball correctness
fast instead of clicking through every file.

Usage:
    python review_output.py --video D:\\Minor_Project\\test_videos\\test_video_1.mp4 \\
                             --masks_dir D:\\Minor_Project\\outputs\\test_video_1 \\
                             --out review_test_video_1.png
"""

import argparse
import glob
import os
import re

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def second_of(path):
    m = re.search(r"mask(\d+)\.png", os.path.basename(path))
    return int(m.group(1))


def get_frame_at_second(cap, fps, second):
    frame_idx = int(second * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--out", default="review.png")
    ap.add_argument("--max_panels", type=int, default=8)
    args = ap.parse_args()

    mask_paths = sorted(glob.glob(os.path.join(args.masks_dir, "mask*.png")), key=second_of)
    if not mask_paths:
        print(f"No mask*.png files found in {args.masks_dir}")
        return

    seconds, has_object = [], []
    for p in mask_paths:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        seconds.append(second_of(p))
        has_object.append(m is not None and m.max() > 0)

    print(f"{args.masks_dir}")
    print(f"  {len(mask_paths)} masks, {sum(has_object)} seconds with a detection")

    ranges = []
    start = None
    for s, h in zip(seconds, has_object):
        if h and start is None:
            start = s
        if not h and start is not None:
            ranges.append((start, s - 1))
            start = None
    if start is not None:
        ranges.append((start, seconds[-1]))

    if ranges:
        print("  Detection ranges (seconds):")
        for a, b in ranges:
            print(f"    {a}s - {b}s")
    else:
        print("  No detections in any second of this video -- either nothing was "
              "flagged, or the object(s) in this video are hard cases (see diagnose.py).")

    # Pick informative panels: start/middle/end of each detection range.
    panel_secs = []
    for a, b in ranges:
        panel_secs.extend(sorted({a, (a + b) // 2, b}))
    panel_secs = sorted(set(panel_secs))
    if not panel_secs:
        # nothing detected -- just sample a spread of seconds so you can confirm
        # the video/masks are at least aligned and not all garbage
        step = max(1, len(seconds) // args.max_panels)
        panel_secs = seconds[::step][:args.max_panels]
    panel_secs = panel_secs[:args.max_panels]

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    n = len(panel_secs)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    if n == 1:
        axes = [axes]

    mask_by_sec = {second_of(p): p for p in mask_paths}
    for ax, sec in zip(axes, panel_secs):
        frame = get_frame_at_second(cap, fps, sec)
        mask = cv2.imread(mask_by_sec[sec], cv2.IMREAD_GRAYSCALE)
        if frame is not None and mask is not None:
            mask_rs = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            overlay = frame.copy()
            overlay[mask_rs > 0] = (0, 0, 255)  # red highlight, BGR
            blended = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
            ax.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        ax.set_title(f"t={sec}s" + (" [obj]" if mask is not None and mask.max() > 0 else ""), fontsize=9)
        ax.axis("off")

    cap.release()
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"  Saved {args.out}")


if __name__ == "__main__":
    main()
