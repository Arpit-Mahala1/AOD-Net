"""
run_pipeline.py

Single entry point that turns a NEW dataset (same folder convention as our
original 26 training videos: sample_XXX folders, each with a video, a
video_XX.txt giving P/C + ground-truth mask intervals, and the maskN.png
files) into a fully evaluated result: data prep, (optional) training,
inference, and a comparison report against that dataset's own ground truth.
This is the "point it at a new dataset and see how well the model does"
pipeline your professor described.

Two modes:
  --model <checkpoint>   Skip training entirely. See how well an EXISTING
                          model (e.g. our best run_sam2_depth4 checkpoint)
                          generalizes to this new dataset out of the box.
                          Data prep is skipped too -- it's only needed for
                          training, not for inference.
  (no --model)            Train a fresh model on this new dataset from
                          scratch using current best-known settings
                          (encoder_depth=4), then evaluate it.

Each stage is a normal call to the existing, already-validated scripts
(dataset_prep.py, train.py, infer.py, evaluate_predictions.py) run as
subprocesses -- nothing here reimplements their logic, it just wires them
together in the right order with the right settings, the same pattern
sam2_generate_labels.py already uses to orchestrate its worker script.

NOTE: this pipeline does NOT run the SAM2 semi-automatic relabeling step --
that needs a human click per object (click_annotate.py) and stays a manual,
one-time step you'd do yourself if you want SAM2-quality labels for a new
dataset. This pipeline always trains on dataset_prep.py's algorithmic labels.

Usage (evaluate our existing best model on a new dataset, no retraining):
    python run_pipeline.py --dataset_root D:\\Someone\\their_dataset ^
                            --out_root D:\\Someone\\pipeline_run ^
                            --model D:\\Minor_Project\\runs\\run_sam2_depth4\\model.pt ^
                            --encoder_depth 4

Usage (train a brand-new model on a new dataset from scratch, then evaluate):
    python run_pipeline.py --dataset_root D:\\Someone\\their_dataset ^
                            --out_root D:\\Someone\\pipeline_run ^
                            --epochs 40 --encoder_depth 4
"""

import argparse
import glob
import os
import subprocess
import sys

from dataset_prep import parse_txt, find_video_file, find_txt_file

HERE = os.path.dirname(os.path.abspath(__file__))


def _check_checkpoint_loads(model_path, encoder_depth, architecture):
    """
    Loads the checkpoint once, up front, against the given encoder_depth AND
    architecture -- so a wrong guess on either fails immediately with one
    clear error, instead of failing partway through a 26-video batch (each
    video re-loading the same mismatched checkpoint and dying the same way,
    one at a time).
    """
    import torch
    import segmentation_models_pytorch as smp
    decoder_channels_by_depth = {5: (256, 128, 64, 32, 16), 4: (256, 128, 64, 32), 3: (256, 128, 64)}
    if architecture == "unet":
        model = smp.Unet(encoder_name="resnet18", encoder_weights=None, encoder_depth=encoder_depth,
                          decoder_channels=decoder_channels_by_depth[encoder_depth], in_channels=6, classes=1)
    else:  # deeplabv3plus
        model = smp.DeepLabV3Plus(encoder_name="resnet18", encoder_weights=None, encoder_depth=encoder_depth,
                                   in_channels=6, classes=1)
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    except RuntimeError as e:
        raise RuntimeError(
            f"Checkpoint {model_path} does NOT match architecture={architecture}, encoder_depth={encoder_depth}. "
            f"Try different values matching what this checkpoint was actually trained "
            f"with. Original error: {e}"
        )


def run(cmd, step_name):
    print(f"\n{'=' * 70}\n[{step_name}] {' '.join(cmd)}\n{'=' * 70}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"[{step_name}] failed with exit code {result.returncode} -- stopping pipeline.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True,
                     help="folder of sample_XXX subfolders, each with a video, a video_XX.txt (P/C + ground-truth "
                          "mask intervals), and maskN.png files -- same convention as our original 26 videos")
    ap.add_argument("--out_root", required=True, help="everything this pipeline produces goes under here")
    ap.add_argument("--model", default=None,
                     help="path to an existing checkpoint to evaluate as-is, skipping data prep and training. "
                          "If omitted, a new model is trained on --dataset_root from scratch first.")
    ap.add_argument("--encoder_depth", type=int, default=None, choices=[3, 4, 5],
                     help="REQUIRED if --model is given -- must match the depth that checkpoint was trained "
                          "with, or loading it will fail with a shape-mismatch error partway through the batch. "
                          "If omitted with --model, this script refuses to guess and stops before running "
                          "anything. If no --model is given (training a new one), defaults to 4.")
    ap.add_argument("--epochs", type=int, default=40, help="only used when training a new model (no --model)")
    ap.add_argument("--frame_stride", type=int, default=3)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--prob_thresh", type=float, default=0.05)
    ap.add_argument("--mask_dilate_px", type=int, default=0)
    ap.add_argument("--disable_flow_check", action="store_true")
    ap.add_argument("--architecture", choices=["unet", "deeplabv3plus"], default=None,
                     help="REQUIRED if --model is given -- must match what that checkpoint was trained with. "
                          "If no --model is given (training a new one), defaults to unet.")
    args = ap.parse_args()

    if args.model and args.architecture is None:
        raise ValueError(
            "--architecture is required when --model is given, alongside --encoder_depth -- same reason: this "
            "script won't guess an architecture that might silently fail to load. Pass --architecture unet or "
            "--architecture deeplabv3plus matching how that checkpoint was trained."
        )
    if args.architecture is None:
        args.architecture = "unet"  # only reached when training a new model from scratch (no --model)

    if args.model and args.encoder_depth is None:
        raise ValueError(
            "--encoder_depth is required when --model is given -- this script won't guess, since a wrong "
            "guess loads silently-mismatched weights and fails partway through the batch instead of upfront. "
            "Check what depth that checkpoint was trained with (look at the train.py command that produced it; "
            "no --encoder_depth flag on that command means it was the default, 5) and pass it explicitly, e.g. "
            "--encoder_depth 5."
        )
    if args.encoder_depth is None:
        args.encoder_depth = 4  # only reached when training a new model from scratch (no --model)

    training_data_dir = os.path.join(args.out_root, "training_data")
    runs_dir = os.path.join(args.out_root, "runs", "pipeline_model")
    predictions_dir = os.path.join(args.out_root, "predictions")
    report_path = os.path.join(args.out_root, "eval_report.csv")
    os.makedirs(args.out_root, exist_ok=True)

    if args.model:
        # Data prep is only needed to produce training data -- if we're not
        # training, skip it. infer.py reads the raw video directly.
        model_path = args.model
        print(f"\n[prepare] Skipped -- no training requested.")
        print(f"[train] Skipped -- evaluating the provided checkpoint as-is: {model_path}")
        print(f"[check] Verifying {model_path} loads at architecture={args.architecture}, encoder_depth={args.encoder_depth} before running any videos...")
        _check_checkpoint_loads(model_path, args.encoder_depth, args.architecture)
        print(f"[check] OK -- checkpoint architecture matches.")
    else:
        run([
            sys.executable, os.path.join(HERE, "dataset_prep.py"),
            "--dataset_root", args.dataset_root, "--out", training_data_dir,
            "--frame_stride", str(args.frame_stride), "--img_size", str(args.img_size),
        ], "prepare")
        run([
            sys.executable, os.path.join(HERE, "train.py"),
            "--data_root", training_data_dir, "--out_dir", runs_dir,
            "--epochs", str(args.epochs), "--encoder_depth", str(args.encoder_depth),
            "--architecture", args.architecture,
        ], "train")
        model_path = os.path.join(runs_dir, "model.pt")

    # ---- Inference, per video, using THAT video's own P/C from its txt
    # file -- these are training-style videos with their own ground truth,
    # not the fixed-P/C official blind test set, so each gets its own P/C. ----
    folders = sorted(glob.glob(os.path.join(args.dataset_root, "sample_*")))
    print(f"\n[infer] Found {len(folders)} video folder(s) to run inference on")
    for folder in folders:
        sample_name = os.path.basename(os.path.normpath(folder))
        try:
            video_path = find_video_file(folder)
            txt_path = find_txt_file(folder, video_path)
            p, c, _ = parse_txt(txt_path)
        except FileNotFoundError as e:
            print(f"[skip] {sample_name}: {e}")
            continue

        out_dir = os.path.join(predictions_dir, sample_name)
        cmd = [
            sys.executable, os.path.join(HERE, "infer.py"),
            "--video", video_path, "--out", out_dir, "--model", model_path,
            "--p", str(p), "--c", str(c), "--img_size", str(args.img_size),
            "--prob_thresh", str(args.prob_thresh), "--encoder_depth", str(args.encoder_depth),
            "--mask_dilate_px", str(args.mask_dilate_px), "--architecture", args.architecture,
        ]
        if args.disable_flow_check:
            cmd.append("--disable_flow_check")
        run(cmd, f"infer:{sample_name}")

    # ---- Evaluate every video against its own ground truth ----
    run([
        sys.executable, os.path.join(HERE, "evaluate_predictions.py"),
        "--dataset_root", args.dataset_root, "--predictions_root", predictions_dir,
        "--out", report_path,
    ], "evaluate")

    print(f"\n{'=' * 70}\nPipeline complete.")
    print(f"Model used:  {model_path}")
    print(f"Predictions: {predictions_dir}")
    print(f"Report:      {report_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
