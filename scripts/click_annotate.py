"""
click_annotate.py

Step 1 of semi-automatic (SAM2-assisted) ground truth generation.

This is the "human-initiated" part your professor is asking for: for each
object in each training video, you look at the frame where it first appears
and click ONCE on it. That single click gets saved to sam2_prompts.json.
sam2_generate_labels.py then feeds these clicks into SAM2, which propagates
each one into a full per-frame mask across the rest of the video -- this
click + auto-propagate workflow is the same one Meta used to build SAM2's
own training data (they call it a "data engine": manual clicks, automatic
temporal propagation, occasional correction).

Run once per OBJECT per video. For a video with two objects (e.g. two bikes
introduced at different times), run this twice with different --second and
--obj_id values.

Usage:
    python click_annotate.py --video D:\\Minor_Project\\dataset\\sample_010\\video_10.mp4 ^
                              --sample_name sample_010 --second 32 --obj_id 1

    REM second object in the same video, e.g. a second bike appearing later:
    python click_annotate.py --video D:\\Minor_Project\\dataset\\sample_018\\video_18.mp4 ^
                              --sample_name sample_018 --second 32 --obj_id 1
    python click_annotate.py --video D:\\Minor_Project\\dataset\\sample_018\\video_18.mp4 ^
                              --sample_name sample_018 --second 74 --obj_id 2
"""

import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use("TkAgg")  # needs an interactive window to click on
import matplotlib.pyplot as plt

PROMPTS_FILE = "sam2_prompts.json"


def load_prompts():
    if os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE) as f:
            return json.load(f)
    return []


def save_prompts(prompts):
    with open(PROMPTS_FILE, "w") as f:
        json.dump(prompts, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--sample_name", required=True, help="must match the dataset folder name, e.g. sample_010")
    ap.add_argument("--second", type=float, required=True, help="the second at which this object is first clearly visible")
    ap.add_argument("--obj_id", type=int, default=1, help="use 1, 2, 3... for multiple objects in the same video")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = int(args.second * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame at {args.second}s from {args.video}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.imshow(frame_rgb)
    ax.set_title(f"{args.sample_name} @ {args.second}s (obj_id={args.obj_id}) -- click ONCE on the object")
    print("A window should have opened. Click once on the object, then close the window if it doesn't close itself.")
    pts = plt.ginput(1, timeout=0)
    plt.close(fig)

    if not pts:
        print("No click received -- nothing saved. Re-run this command.")
        return

    x, y = pts[0]
    prompts = load_prompts()
    prompts.append({
        "sample_name": args.sample_name,
        "video": args.video,
        "frame_idx": frame_idx,
        "obj_id": args.obj_id,
        "x": float(x),
        "y": float(y),
    })
    save_prompts(prompts)
    print(f"Saved click at pixel ({x:.0f}, {y:.0f}) on frame {frame_idx} ({args.second}s) "
          f"for {args.sample_name}, obj_id={args.obj_id}")
    print(f"Total prompts saved so far (across all videos): {len(prompts)}")


if __name__ == "__main__":
    main()
