# AOD-Net

Abandoned object detection for fixed-camera videos. You give it a video and two
times, P and C (in seconds), and it saves one mask image per second showing
objects that were left behind and stayed in place.

Paper details will be added here after publication.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Training needs a CUDA GPU. Running on a video also works on CPU, just slowly.

## Weights

The trained model is not stored in the repo. Download `aod_net_final.pt` from the
Releases page, create a `weights/` folder and put the file in it.

## Run on a video

```
python scripts/infer.py --video my_video.mp4 --out outputs/my_video --model weights/aod_net_final.pt --p 60 --c 90 --architecture unet --encoder_depth 5
```

- `--p`: how many seconds an object must stay still before it is reported
- `--c`: how many seconds it keeps being reported after that

No labels are needed for this step.

## Train and evaluate

This needs your own labelled data, which is not included. Each video goes in its
own folder:

```
dataset/
  sample_001/
    video_1.mp4
    video_1.txt
    mask2.png
```

The txt file has P, C and lines like `mask2.png: 65s-109s`. The mask files are the
object shapes those lines point to.

```
python scripts/dataset_prep.py --dataset_root dataset --out training_data
python scripts/train.py --data_root training_data --epochs 40 --out_dir runs/my_run --seed 42 --disable_amp
python scripts/run_pipeline.py --dataset_root dataset --out_root pipeline_out --model runs/my_run/model.pt --architecture unet --encoder_depth 5
```

`run_pipeline.py` needs `--architecture` and `--encoder_depth` to match the
checkpoint. If they don't, it stops before running.

## Scripts

| Script | What it does |
|---|---|
| `dataset_prep.py` | Turns videos and mask intervals into per-frame training data |
| `train.py` | Trains the model |
| `infer.py` | Runs a trained model on a video and writes the per-second masks |
| `evaluate_predictions.py` | Compares predicted masks with ground truth |
| `run_pipeline.py` | Runs infer and evaluate on a whole dataset in one go |
| `diagnose.py` | Shows raw per-pixel scores for chosen frames |
| `review_output.py` | Makes an overlay image of the detections |
| `check_split.py` | Prints which videos are in train and validation |
| `yolo_prepare_data.py`, `yolo_evaluate.py` | Same-split comparison with YOLO |
| `click_annotate.py`, `sam2_generate_*.py` | Optional SAM2-assisted labeling |