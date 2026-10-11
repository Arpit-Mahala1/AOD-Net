"""
dataset_prep.py

Converts your 26 (video + interval-masks + P/C text file) folders into a dense
per-frame training set: (current_frame, background_reference) -> raw_object_mask

Key idea: the raw label is NOT the P/C-gated mask. It's just "is the object
physically present and static here" -- true from the moment it appears until
the end of the video (or until it's removed, if you extend this later).

appearance_time is computed analytically from the text file:
    appearance_time = (start of the first non-black ground-truth mask) - P
This matches the definition your professor's example uses (35s + 30s = 65s),
so no extra object detection is needed to find it.

Usage:
    python dataset_prep.py --dataset_root D:\\Minor_Project\\dataset --out D:\\Minor_Project\\training_data
"""

import argparse
import glob
import os
import re

import cv2
import numpy as np

MASK_LINE_RE = re.compile(r"(mask\d+\.png)\s*:.*?(\d+)s\s*(?:[–—-]|to)\s*(\d+)s", re.IGNORECASE)
P_RE = re.compile(r"Persistence Threshold\s*\(P\)\s*:\s*(\d+)")
C_RE = re.compile(r"Cooldown Period\s*\(C\)\s*:\s*(\d+)")


def parse_txt(txt_path):
    text = open(txt_path, "r", encoding="utf-8", errors="ignore").read()
    p = int(P_RE.search(text).group(1))
    c = int(C_RE.search(text).group(1))
    intervals = []
    for m in MASK_LINE_RE.finditer(text):
        fname, start, end = m.group(1), int(m.group(2)), int(m.group(3))
        intervals.append((fname, start, end))
    return p, c, intervals


def find_video_file(folder):
    for ext in ("mp4", "avi", "mov", "mkv"):
        matches = glob.glob(os.path.join(folder, f"video_*.{ext}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No video_*.* file found in {folder}")


def find_txt_file(folder, video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]  # video_01
    candidate = os.path.join(folder, stem + ".txt")
    if os.path.exists(candidate):
        return candidate
    matches = glob.glob(os.path.join(folder, "*.txt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No txt file found in {folder}")


def extract_object_tracks(folder, intervals, min_new_area=300, dilate_px=4):
    """
    A video can have multiple independently-persisting objects, and objects
    that are close together can appear FUSED into a single connected blob in
    a mask that flags both at once (e.g. two bikes parked near each other:
    mask2 shows bike-1 alone, mask3 shows bike-1 + bike-2 as one touching
    blob, mask4 shows bike-2 alone). Whole-blob matching can't split that
    fused blob back into two objects.

    Instead: process masks in chronological order and keep a running union of
    already-known object shapes. In each new mask, only the pixels NOT
    already accounted for are candidates for a new object. Because a
    stationary object sits in the exact same pixels every time it's flagged,
    subtracting known shapes cleanly isolates a newly-revealed object even
    when it shares a merged blob with an object we already know about.

    Returns one track per distinct object: {"mask": canonical shape,
    "first_flagged_start": earliest second it was flagged}.
    """
    mask_entries = []
    target_shape = None  # (h, w) -- some datasets have off-by-a-few-pixel mask files
    for fname, start, end in intervals:
        mpath = os.path.join(folder, fname)
        if not os.path.exists(mpath):
            continue
        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if m is None or m.max() == 0:
            continue
        if target_shape is None:
            target_shape = m.shape
        elif m.shape != target_shape:
            m = cv2.resize(m, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        binary = (m > 0).astype(np.uint8) * 255
        mask_entries.append((start, binary))
    mask_entries.sort(key=lambda e: e[0])

    tracks = []
    known_union = None
    kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)

    for start, binary in mask_entries:
        if known_union is None:
            known_union = np.zeros_like(binary)

        # Slightly dilate what we already know before subtracting, so minor
        # rendering/anti-aliasing differences between mask files don't leave
        # a spurious thin "rim" that gets picked up as a fake new object.
        known_dilated = cv2.dilate(known_union, kernel)
        new_regions = cv2.bitwise_and(binary, cv2.bitwise_not(known_dilated))
        new_regions = cv2.morphologyEx(new_regions, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(new_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_new_area:
                continue
            blob_mask = np.zeros_like(binary)
            cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
            tracks.append({"mask": blob_mask, "first_flagged_start": start})
            known_union = cv2.bitwise_or(known_union, blob_mask)

    return tracks


class RollingBackground:
    """Same causal rolling-median background model used at inference time."""

    def __init__(self, history_frames):
        self.history_frames = max(1, history_frames)
        self.buffer = []
        self.bg = None

    def update(self, frame_bgr):
        self.buffer.append(frame_bgr)
        if len(self.buffer) > self.history_frames:
            self.buffer.pop(0)
        self.bg = np.median(np.stack(self.buffer, axis=0), axis=0).astype(np.uint8)
        return self.bg


def already_processed(sample_name, out_dir):
    return len(glob.glob(os.path.join(out_dir, "frames", f"{sample_name}_*.png"))) > 0


def process_folder(folder, out_dir, frame_stride=3, img_size=256, force=False):
    sample_name = os.path.basename(os.path.normpath(folder))

    if not force and already_processed(sample_name, out_dir):
        print(f"[skip - already done] {folder}")
        return 0

    video_path = find_video_file(folder)
    txt_path = find_txt_file(folder, video_path)
    p, c, intervals = parse_txt(txt_path)

    tracks = extract_object_tracks(folder, intervals)
    if not tracks:
        print(f"[skip] {folder}: no non-black ground-truth mask found")
        return 0

    objects = []
    for t in tracks:
        appearance_time = max(0, t["first_flagged_start"] - p)
        shape_rs = cv2.resize(t["mask"], (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        objects.append({"shape": shape_rs, "appearance_time": appearance_time})

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    bg_model = RollingBackground(history_frames=int(fps * 3))

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize down to img_size FIRST, then run the rolling background
        # median on the small frame. Running the median at full video
        # resolution on every single frame (even ones we don't save) was
        # the actual bottleneck behind the original 10-12 hour run.
        frame_rs = cv2.resize(frame, (img_size, img_size))
        bg_rs = bg_model.update(frame_rs)
        t_sec = frame_idx / fps

        if frame_idx % frame_stride == 0 and len(bg_model.buffer) >= min(5, bg_model.history_frames):
            # Raw label = union of every object physically present at this
            # moment, regardless of P/C -- an object counts as "present" from
            # its own individual appearance time onward.
            label = np.zeros((img_size, img_size), dtype=np.uint8)
            for obj in objects:
                if t_sec >= obj["appearance_time"]:
                    label = cv2.bitwise_or(label, obj["shape"])

            base = f"{sample_name}_{frame_idx:06d}"
            cv2.imwrite(os.path.join(out_dir, "frames", base + ".png"), frame_rs)
            cv2.imwrite(os.path.join(out_dir, "backgrounds", base + ".png"), bg_rs)
            cv2.imwrite(os.path.join(out_dir, "labels", base + ".png"), label)
            saved += 1

        frame_idx += 1

    cap.release()
    obj_summary = ", ".join(f"obj{i+1}@{o['appearance_time']:.0f}s" for i, o in enumerate(objects))
    print(f"[ok] {folder}: {len(objects)} object(s) [{obj_summary}], p={p}, c={c}, saved {saved} frames")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, help="folder containing sample_001, sample_002, ...")
    ap.add_argument("--out", required=True, help="output training_data folder")
    ap.add_argument("--frame_stride", type=int, default=3)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--samples", default=None,
                     help="comma-separated sample folder names to (re)process, e.g. sample_003,sample_004,sample_014. "
                          "Default: all folders, auto-skipping any already processed.")
    ap.add_argument("--force", action="store_true",
                     help="reprocess even if frames already exist for that sample (deletes+rewrites its frames)")
    args = ap.parse_args()

    for sub in ("frames", "backgrounds", "labels"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    all_folders = sorted(glob.glob(os.path.join(args.dataset_root, "sample_*")))

    if args.samples:
        wanted = {s.strip() for s in args.samples.split(",")}
        folders = [f for f in all_folders if os.path.basename(os.path.normpath(f)) in wanted]
        missing = wanted - {os.path.basename(os.path.normpath(f)) for f in folders}
        if missing:
            print(f"WARNING: requested but not found under --dataset_root: {sorted(missing)}")
    else:
        folders = all_folders

    print(f"Found {len(all_folders)} sample folders total, processing {len(folders)}")

    total = 0
    for folder in folders:
        total += process_folder(folder, args.out, args.frame_stride, args.img_size, force=args.force)

    print(f"\nTotal NEW training frames saved this run: {total}")


if __name__ == "__main__":
    main()
