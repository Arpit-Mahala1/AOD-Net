"""
sam2_generate_one_sample.py

Worker script -- processes exactly ONE sample (one video). Meant to be
launched as its own separate OS process by sam2_generate_labels.py, not run
directly for the whole dataset.

Why a separate process per sample: a genuine CUDA error (not just
out-of-memory, but context corruption like CUBLAS_STATUS_INTERNAL_ERROR) can
escalate into a hard native abort that kills the entire Python interpreter --
this bypasses try/except completely, since it happens below the Python
layer. Isolating each sample in its own process means a hard crash on one
video only kills that one process; the orchestrator and every other sample
are unaffected, each starting with a completely fresh CUDA context.

Labels are also written INCREMENTALLY, per chunk, as soon as each chunk's
masks are computed (read-modify-write union into the output PNG) -- not
batched up and written only at the very end. So even if THIS process crashes
partway through, whatever objects/chunks finished before the crash are
already safely on disk.

You normally don't call this directly -- run sam2_generate_labels.py, which
launches this once per sample_name found in sam2_prompts.json.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

PROMPTS_FILE = "sam2_prompts.json"


def get_saved_frame_indices(training_data, sample_name):
    paths = sorted(glob.glob(os.path.join(training_data, "frames", f"{sample_name}_*.png")))
    return sorted(int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]) for p in paths)


def extract_specific_frames(video_path, frame_indices, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    frame_indices = sorted(frame_indices)
    wanted = set(frame_indices)
    pos_in_list = {orig_idx: i for i, orig_idx in enumerate(frame_indices)}
    cap = cv2.VideoCapture(video_path)
    idx = 0
    saved = 0
    while saved < len(frame_indices):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in wanted:
            cv2.imwrite(os.path.join(out_dir, f"{pos_in_list[idx]:06d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved


def mask_to_box(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def update_label_file(out_labels_dir, sample_name, orig_idx, mask_native, img_size):
    """Read-modify-write union: adds this object's mask into whatever is
    already saved for this frame, instead of overwriting it. This is what
    lets multiple objects (processed one after another) safely combine, and
    what makes per-chunk incremental writing safe to resume after a crash."""
    path = os.path.join(out_labels_dir, f"{sample_name}_{orig_idx:06d}.png")
    mask_u8 = (mask_native.astype(np.uint8)) * 255
    mask_rs = cv2.resize(mask_u8, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    if os.path.exists(path):
        existing = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if existing is not None and existing.shape == mask_rs.shape:
            mask_rs = cv2.bitwise_or(existing, mask_rs)
    cv2.imwrite(path, mask_rs)


def propagate_object_in_chunks_and_write(predictor, device, autocast_dtype, frames_dir, extracted_indices,
                                          click_original_idx, click_xy, obj_id, chunk_frames,
                                          saved_frame_set, out_labels_dir, sample_name, img_size):
    click_pos = extracted_indices.index(click_original_idx)
    total = len(extracted_indices)
    chunk_start = click_pos
    box_prompt = None
    x, y = click_xy
    n_written = 0

    while chunk_start < total:
        chunk_end = min(chunk_start + chunk_frames, total)
        chunk_dir = tempfile.mkdtemp()
        try:
            for rel_i, src_pos in enumerate(range(chunk_start, chunk_end)):
                shutil.copy(
                    os.path.join(frames_dir, f"{src_pos:06d}.jpg"),
                    os.path.join(chunk_dir, f"{rel_i:06d}.jpg"),
                )

            with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
                state = predictor.init_state(chunk_dir, offload_video_to_cpu=True, offload_state_to_cpu=True)
                if box_prompt is None:
                    predictor.add_new_points_or_box(
                        state, frame_idx=0, obj_id=obj_id,
                        points=np.array([[x, y]], dtype=np.float32),
                        labels=np.array([1], dtype=np.int32),
                    )
                else:
                    predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_id, box=box_prompt)

                last_mask = None
                for rel_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                    m = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
                    orig_idx = extracted_indices[chunk_start + rel_idx]
                    if orig_idx in saved_frame_set:
                        update_label_file(out_labels_dir, sample_name, orig_idx, m, img_size)
                        n_written += 1
                    last_mask = m

                predictor.reset_state(state)
            del state
        finally:
            shutil.rmtree(chunk_dir, ignore_errors=True)
            if device == "cuda":
                torch.cuda.empty_cache()

        box_prompt = mask_to_box(last_mask) if last_mask is not None else None
        if box_prompt is None:
            print(f"    [warn] obj_id={obj_id} lost track after original frame "
                  f"{extracted_indices[chunk_end - 1]} -- stopping this object's propagation early.")
            break
        chunk_start = chunk_end

    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_name", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--training_data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="facebook/sam2.1-hiera-tiny")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--chunk_frames", type=int, default=100)
    args = ap.parse_args()

    with open(PROMPTS_FILE) as f:
        all_prompts = json.load(f)
    sample_prompts = [p for p in all_prompts if p["sample_name"] == args.sample_name]
    if not sample_prompts:
        print(f"No prompts found for {args.sample_name} in {PROMPTS_FILE}")
        sys.exit(1)

    video_path = sample_prompts[0]["video"]
    saved_frame_indices = get_saved_frame_indices(args.training_data, args.sample_name)
    if not saved_frame_indices:
        print(f"No frames found for {args.sample_name} under {args.training_data} -- did dataset_prep.py process it?")
        sys.exit(1)
    saved_frame_set = set(saved_frame_indices)

    earliest_click = min(p["frame_idx"] for p in sample_prompts)
    needed = {f for f in saved_frame_indices if f >= earliest_click}
    needed |= {p["frame_idx"] for p in sample_prompts}
    extracted_indices = sorted(needed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[{args.sample_name}] device={device} checkpoint={args.checkpoint} chunk_frames={args.chunk_frames}")
    predictor = SAM2VideoPredictor.from_pretrained(args.checkpoint)

    out_labels_dir = os.path.join(args.out, "labels")
    os.makedirs(out_labels_dir, exist_ok=True)

    frames_dir = tempfile.mkdtemp()
    try:
        n = extract_specific_frames(video_path, extracted_indices, frames_dir)
        print(f"[{args.sample_name}] extracted {n} needed frames (of {len(saved_frame_indices)} kept by dataset_prep.py)")

        for p in sample_prompts:
            written = propagate_object_in_chunks_and_write(
                predictor, device, autocast_dtype, frames_dir, extracted_indices,
                p["frame_idx"], (p["x"], p["y"]), p["obj_id"], args.chunk_frames,
                saved_frame_set, out_labels_dir, args.sample_name, args.img_size,
            )
            print(f"[{args.sample_name}] obj_id={p['obj_id']}: wrote/updated {written} label frames")
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)

    # Any saved frame before the earliest click, or never touched by any
    # object, should be black -- write zeros for anything still missing.
    filled = 0
    for orig_idx in saved_frame_indices:
        path = os.path.join(out_labels_dir, f"{args.sample_name}_{orig_idx:06d}.png")
        if not os.path.exists(path):
            cv2.imwrite(path, np.zeros((args.img_size, args.img_size), dtype=np.uint8))
            filled += 1
    print(f"[{args.sample_name}] filled {filled} pre-appearance/untouched frames with black")
    print(f"[{args.sample_name}] DONE")


if __name__ == "__main__":
    main()
