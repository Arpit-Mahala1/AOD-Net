"""
sam2_generate_labels.py

Orchestrator -- reads sam2_prompts.json, finds every unique sample_name in
it, and runs sam2_generate_one_sample.py as a SEPARATE PROCESS for each one.

Why a subprocess per sample instead of just a loop in this same process: a
genuine CUDA context error (not just out-of-memory, but something like
CUBLAS_STATUS_INTERNAL_ERROR) can escalate into a hard native crash that
kills the whole Python interpreter -- this cannot be caught by try/except,
since it happens below the Python layer. Running each sample in its own
process means that if one crashes, only that process dies; this orchestrator
and every subsequent sample get a completely fresh CUDA context and are
unaffected.

Usage (same command as before -- this now delegates the real work):
    python sam2_generate_labels.py --dataset_root D:\\Minor_Project\\dataset ^
                                    --training_data D:\\Minor_Project\\training_data ^
                                    --out D:\\Minor_Project\\training_data_sam2 ^
                                    --checkpoint facebook/sam2.1-hiera-tiny ^
                                    --chunk_frames 100
"""

import argparse
import json
import os
import subprocess
import sys

PROMPTS_FILE = "sam2_prompts.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--training_data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="facebook/sam2.1-hiera-tiny")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--chunk_frames", type=int, default=100)
    args = ap.parse_args()

    if not os.path.exists(PROMPTS_FILE):
        raise FileNotFoundError(f"{PROMPTS_FILE} not found -- run click_annotate.py first for each object/video.")

    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)
    sample_names = sorted({p["sample_name"] for p in prompts})
    print(f"Found {len(sample_names)} sample(s) with prompts: {sample_names}")

    worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2_generate_one_sample.py")

    results = {}
    for sample_name in sample_names:
        print(f"\n{'=' * 60}\nRunning {sample_name} in its own process...\n{'=' * 60}")
        cmd = [
            sys.executable, worker_path,
            "--sample_name", sample_name,
            "--dataset_root", args.dataset_root,
            "--training_data", args.training_data,
            "--out", args.out,
            "--checkpoint", args.checkpoint,
            "--img_size", str(args.img_size),
            "--chunk_frames", str(args.chunk_frames),
        ]
        # No output capturing -- let the worker's prints/progress bars show
        # live in this same console, exactly as before.
        proc = subprocess.run(cmd)
        results[sample_name] = proc.returncode
        if proc.returncode != 0:
            print(f"\n[ORCHESTRATOR] {sample_name} exited with code {proc.returncode} "
                  f"(likely crashed) -- moving on to the next sample. Labels already "
                  f"written for {sample_name} before the crash are NOT lost (written incrementally).")

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    ok = [s for s, rc in results.items() if rc == 0]
    failed = [s for s, rc in results.items() if rc != 0]
    print(f"Succeeded ({len(ok)}): {ok}")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")
        print("For a failed sample, try re-running just that one with a smaller --chunk_frames, e.g.:")
        if failed:
            print(f"  python sam2_generate_one_sample.py --sample_name {failed[0]} "
                  f"--dataset_root {args.dataset_root} --training_data {args.training_data} "
                  f"--out {args.out} --checkpoint {args.checkpoint} --chunk_frames 50")
    else:
        print("All samples completed.")

    print(f"\nLabels written to {os.path.join(args.out, 'labels')}")
    print("Next: copy the ORIGINAL 'frames' and 'backgrounds' folders from your training_data "
          f"into {args.out}, then point train.py --data_root at {args.out}")


if __name__ == "__main__":
    main()
