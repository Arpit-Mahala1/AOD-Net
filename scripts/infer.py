"""
infer.py

Runs the trained per-frame model on test videos and applies the P/C business
rule as a deterministic post-processing step (no learning here -- just a
per-blob timer), producing one mask PNG per second.

The model only ever answers "is there a static foreign object here right now."
Turning that into "has it been there for at least P seconds, and less than
P+C seconds" is plain bookkeeping, tracked per connected component across
frames using simple centroid matching (Hungarian algorithm).

P=60, C=90 for all test videos by default -- no per-video txt file needed.

Usage (single video):
    python infer.py --video D:\\Minor_Project\\test_videos\\test_video_1.mp4 \\
                     --model model.pt --out D:\\Minor_Project\\outputs\\test_video_1

Usage (batch -- all videos sitting directly in one folder, no subfolders):
    python infer.py --video_dir D:\\Minor_Project\\test_videos \\
                     --model model.pt --out_root D:\\Minor_Project\\outputs
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import segmentation_models_pytorch as smp

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

VIDEO_EXTS = ("mp4", "avi", "mov", "mkv")


def normalize(img_bgr, size):
    img = cv2.resize(img_bgr, (size, size))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return rgb.transpose(2, 0, 1)


class RollingBackground:
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


class BlobTimer:
    """Tracks connected components across frames to know how long each has existed.

    Also keeps a rolling optical-flow "stayed still" history per track. This
    is a second, independent stationarity check on top of the P/C age gate:
    age tells you a blob has existed for a while, flow tells you it hasn't
    actually been moving (drifting shadow, lighting flicker the background
    model mistakes for a static object) during that time. A track only
    counts as flow-confirmed-static once it has accumulated enough history
    (flow_min_frames) -- before that it defaults to "pass" so brand-new
    tracks aren't rejected the instant they're created.
    """

    def __init__(self, max_match_dist=45, max_missed_frames=15,
                 flow_static_thresh=0.5, flow_min_frames=10,
                 flow_min_static_frac=0.8, flow_history_cap=300):
        self.tracks = {}
        self.next_id = 0
        self.max_match_dist = max_match_dist
        self.max_missed_frames = max_missed_frames
        self.flow_static_thresh = flow_static_thresh
        self.flow_min_frames = flow_min_frames
        self.flow_min_static_frac = flow_min_static_frac
        self.flow_history_cap = flow_history_cap  # cap so ancient history doesn't dilute recent motion

    def step(self, blobs, frame_idx, flow_mag=None):
        track_ids = list(self.tracks.keys())
        matched_blobs = set()

        if track_ids and blobs:
            cost = np.zeros((len(track_ids), len(blobs)))
            for i, tid in enumerate(track_ids):
                tc = self.tracks[tid]["centroid"]
                for j, b in enumerate(blobs):
                    bc = b["centroid"]
                    cost[i, j] = np.hypot(tc[0] - bc[0], tc[1] - bc[1])
            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] > self.max_match_dist:
                    continue
                tid, blob = track_ids[r], blobs[c]
                t = self.tracks[tid]
                t["centroid"] = blob["centroid"]
                t["mask"] = blob["mask"]
                t["last_seen_frame"] = frame_idx
                self._update_flow_history(t, blob["mask"], flow_mag)
                matched_blobs.add(c)

        for j, b in enumerate(blobs):
            if j not in matched_blobs:
                self.tracks[self.next_id] = {
                    "centroid": b["centroid"], "mask": b["mask"],
                    "first_seen_frame": frame_idx, "last_seen_frame": frame_idx,
                    "flow_history": [],
                }
                # Deliberately no flow sample on the creation frame: flow_mag
                # is prev_frame->this_frame, and a track that didn't exist in
                # prev_frame has no meaningful "did it move" answer yet.
                self.next_id += 1

        stale = [tid for tid, t in self.tracks.items()
                 if frame_idx - t["last_seen_frame"] > self.max_missed_frames]
        for tid in stale:
            del self.tracks[tid]

        return self.tracks

    def _update_flow_history(self, track, mask, flow_mag):
        if flow_mag is None:
            return
        region = flow_mag[mask > 0]
        if region.size == 0:
            return
        is_static_this_frame = bool(region.mean() < self.flow_static_thresh)
        history = track["flow_history"]
        history.append(is_static_this_frame)
        if len(history) > self.flow_history_cap:
            del history[: len(history) - self.flow_history_cap]

    def flow_confirmed_static(self, track):
        """True if this track passes the flow check, or hasn't accumulated
        enough history yet to be judged (defaults to pass, not fail)."""
        history = track["flow_history"]
        if len(history) < self.flow_min_frames:
            return True
        return (sum(history) / len(history)) >= self.flow_min_static_frac


def extract_blobs(binary_mask, min_area=100, dilate_px=0):
    if dilate_px > 0:
        kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        binary_mask = cv2.dilate(binary_mask, kernel)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        mask = np.zeros_like(binary_mask)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        blobs.append({"centroid": (x + w / 2, y + h / 2), "mask": mask})
    return blobs


def run_on_video(video_path, model, device, out_dir, p, c, img_size=256, prob_thresh=0.5,
                  flow_static_thresh=0.5, flow_min_frames=10, flow_min_static_frac=0.8,
                  use_flow_check=True, mask_dilate_px=0):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg_model = RollingBackground(history_frames=int(fps * 3))
    timer = BlobTimer(flow_static_thresh=flow_static_thresh, flow_min_frames=flow_min_frames,
                       flow_min_static_frac=flow_min_static_frac)

    frame_idx = 0
    last_second_written = -1
    prev_gray = None

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize down to model resolution FIRST. Everything after this
            # point (background median, model inference, blob tracking, flow)
            # runs at img_size, not the original video resolution. Doing the
            # rolling median at full HD resolution every frame was the actual
            # bottleneck -- this is what made a several-hour run not finish.
            frame_small = cv2.resize(frame, (img_size, img_size))
            bg_small = bg_model.update(frame_small)

            x_frame = normalize(frame_small, img_size)
            x_bg = normalize(bg_small, img_size)
            x = np.concatenate([x_frame, x_bg], axis=0)[None, ...]
            x_t = torch.from_numpy(x).to(device)

            logits = model(x_t)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            binary_small = (probs > prob_thresh).astype(np.uint8) * 255

            blobs = extract_blobs(binary_small, dilate_px=mask_dilate_px)  # stays at img_size, cheap

            flow_mag = None
            if use_flow_check:
                gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                    )
                    flow_mag = np.hypot(flow[..., 0], flow[..., 1])
                prev_gray = gray

            tracks = timer.step(blobs, frame_idx, flow_mag=flow_mag)

            second = int(frame_idx / fps)
            if second != last_second_written:
                out_mask_small = np.zeros((img_size, img_size), dtype=np.uint8)
                for t in tracks.values():
                    age_secs = (frame_idx - t["first_seen_frame"]) / fps
                    age_ok = p <= age_secs < (p + c)
                    flow_ok = (not use_flow_check) or timer.flow_confirmed_static(t)
                    if age_ok and flow_ok:
                        out_mask_small = cv2.bitwise_or(out_mask_small, t["mask"])
                # Upscale to original video resolution only here -- once per
                # second, not once per frame -- for the saved output mask.
                out_mask = cv2.resize(out_mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(out_dir, f"mask{second + 1}.png"), out_mask)
                last_second_written = second

            frame_idx += 1

    cap.release()
    print(f"  -> {out_dir} ({last_second_written + 1} masks)")


def find_videos(video_dir):
    found = []
    for ext in VIDEO_EXTS:
        found.extend(glob.glob(os.path.join(video_dir, f"*.{ext}")))
    return sorted(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="single video file")
    ap.add_argument("--out", help="output dir for --video mode")
    ap.add_argument("--video_dir", help="batch mode: folder containing video files directly (no subfolders/txt needed)")
    ap.add_argument("--out_root", help="batch mode: parent output dir; one subfolder per video is created automatically")
    ap.add_argument("--model", required=True)
    ap.add_argument("--p", type=int, default=60, help="Persistence Threshold in seconds (default 60)")
    ap.add_argument("--c", type=int, default=90, help="Cooldown Period in seconds (default 90)")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--prob_thresh", type=float, default=0.5)
    ap.add_argument("--encoder_depth", type=int, default=5, choices=[3, 4, 5],
                     help="MUST match the value used when this checkpoint was trained")
    ap.add_argument("--architecture", choices=["unet", "deeplabv3plus"], default="unet",
                     help="MUST match what this checkpoint was trained with (train.py's --architecture)")
    ap.add_argument("--flow_static_thresh", type=float, default=0.5,
                     help="mean optical-flow magnitude (px/frame, at img_size resolution) inside a "
                          "blob's mask below which that frame counts as 'stayed still'")
    ap.add_argument("--flow_min_frames", type=int, default=10,
                     help="a track needs at least this many flow samples before the flow gate applies; "
                          "below this it defaults to pass, so new tracks aren't rejected on arrival")
    ap.add_argument("--flow_min_static_frac", type=float, default=0.8,
                     help="fraction of a track's flow history that must be 'stayed still' to pass the gate")
    ap.add_argument("--disable_flow_check", action="store_true",
                     help="turn off the optical-flow gate entirely and fall back to pure P/C age gating "
                          "(useful for A/B comparing against the old behavior on the same checkpoint)")
    ap.add_argument("--mask_dilate_px", type=int, default=0,
                     help="grow each predicted blob mask by this many pixels (at img_size resolution) before "
                          "tracking -- a cheap, no-retrain way to recover coverage when the model finds the "
                          "right location but undersizes the object's extent. 0 = off (unchanged behavior).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    flow_status = "off (--disable_flow_check)" if args.disable_flow_check else \
        f"on (thresh={args.flow_static_thresh}, min_frames={args.flow_min_frames}, min_static_frac={args.flow_min_static_frac})"
    print(f"Using device: {device}  |  P={args.p}s  C={args.c}s  |  flow check: {flow_status}")

    decoder_channels_by_depth = {5: (256, 128, 64, 32, 16), 4: (256, 128, 64, 32), 3: (256, 128, 64)}
    if args.architecture == "unet":
        model = smp.Unet(encoder_name="resnet18", encoder_weights=None, encoder_depth=args.encoder_depth,
                          decoder_channels=decoder_channels_by_depth[args.encoder_depth], in_channels=6, classes=1)
    else:  # deeplabv3plus
        model = smp.DeepLabV3Plus(encoder_name="resnet18", encoder_weights=None, encoder_depth=args.encoder_depth,
                                   in_channels=6, classes=1)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.to(device).eval()

    if args.video_dir:
        if not args.out_root:
            raise ValueError("--out_root is required with --video_dir")
        videos = find_videos(args.video_dir)
        print(f"Found {len(videos)} video(s) in {args.video_dir}")
        for video_path in videos:
            stem = os.path.splitext(os.path.basename(video_path))[0]
            out_dir = os.path.join(args.out_root, stem)
            print(f"Processing {video_path}")
            run_on_video(video_path, model, device, out_dir, args.p, args.c, args.img_size, args.prob_thresh,
                         flow_static_thresh=args.flow_static_thresh, flow_min_frames=args.flow_min_frames,
                         flow_min_static_frac=args.flow_min_static_frac,
                         use_flow_check=not args.disable_flow_check, mask_dilate_px=args.mask_dilate_px)
    elif args.video:
        if not args.out:
            raise ValueError("--out is required with --video")
        run_on_video(args.video, model, device, args.out, args.p, args.c, args.img_size, args.prob_thresh,
                     flow_static_thresh=args.flow_static_thresh, flow_min_frames=args.flow_min_frames,
                     flow_min_static_frac=args.flow_min_static_frac,
                     use_flow_check=not args.disable_flow_check, mask_dilate_px=args.mask_dilate_px)
    else:
        raise ValueError("Provide either --video/--out or --video_dir/--out_root")

    print("Done.")


if __name__ == "__main__":
    main()

