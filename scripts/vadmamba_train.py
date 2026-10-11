"""
vadmamba_train.py

Trains a frame-prediction anomaly detector: given K consecutive NORMAL
frames (object not yet present, per your labels/*.png masks), predict the
next frame. Trained ONLY on runs of consecutive normal frames within each
train-split video (never crosses from one video into another, and never
includes a frame after the object has appeared).

At inference (vadmamba_evaluate.py), the same predictor is run across real
val-video sequences -- when the object is present, the model has never
learned what that looks like, so prediction error should spike there. That
per-pixel error map, thresholded, becomes the predicted "anomaly mask" --
scored with the same TP/FP/FN accounting as every other baseline.

See vadmamba_prepare_data.py's docstring for what "VADMamba-inspired" means
here and why (official code needs two Windows-hostile CUDA dependencies).

Architecture: the same conv-encoder / Mamba-bottleneck / conv-decoder shape
as mamba_unet_train.py, but re-purposed for prediction instead of
segmentation: input is K stacked RGB frames (3*K channels), output is a
single predicted RGB frame (3 channels), trained with L1 reconstruction
loss. No optical-flow branch (VADMamba's second task) is included -- purely
appearance-based prediction, which is a real simplification worth stating
in your write-up.

Install (in addition to torch/torchvision and mambapy, which you already
have from the mamba_unet baseline):
    (nothing new)

Usage:
    python vadmamba_train.py --vadmamba_dataset D:\\Minor_Project\\vadmamba_dataset ^
                              --epochs 40 --out_dir D:\\Minor_Project\\runs\\vadmamba
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from mambapy.mamba import Mamba, MambaConfig

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)


def denormalize(x):
    """x: C,H,W normalized -> H,W,C in [0,1], for computing pixel error in a stable range."""
    img = x.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


class NormalRunsDataset(Dataset):
    """Builds (K frames -> next frame) training windows from runs of
    consecutive normal frames within each video, per the manifest."""

    def __init__(self, images_dir, manifest_path, k=4):
        self.images_dir = images_dir
        self.k = k
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.windows = []  # list of (list of k frame paths, target frame path)
        for video, frames in manifest.items():
            run = []
            for entry in frames:
                if entry["normal"]:
                    run.append(entry["frame"])
                else:
                    run = []  # break the run -- never train across an anomalous frame
                if len(run) > k:
                    self.windows.append((run[-k - 1:-1], run[-1]))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ctx_names, target_name = self.windows[idx]
        ctx = [normalize(cv2.imread(os.path.join(self.images_dir, n))) for n in ctx_names]
        target = normalize(cv2.imread(os.path.join(self.images_dir, target_name)))
        x = np.concatenate(ctx, axis=0)  # 3*K, H, W
        return torch.from_numpy(x), torch.from_numpy(target)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MambaPredictor(nn.Module):
    """Same conv+Mamba-bottleneck U-Net shape as mamba_unet_train.py's
    MambaUNet, but K*3 channels in, 3 channels out (predicted next frame)."""

    def __init__(self, k=4, n_mamba_layers=4, d_state=16):
        super().__init__()
        ch = [32, 64, 128, 256, 512]
        self.stem = ConvBlock(3 * k, ch[0])
        self.down1 = Down(ch[0], ch[1])
        self.down2 = Down(ch[1], ch[2])
        self.down3 = Down(ch[2], ch[3])
        self.down4 = Down(ch[3], ch[4])

        self.mamba = Mamba(MambaConfig(d_model=ch[4], n_layers=n_mamba_layers, d_state=d_state))
        self.bottleneck_norm = nn.LayerNorm(ch[4])

        self.up1 = Up(ch[4], ch[3], ch[3])
        self.up2 = Up(ch[3], ch[2], ch[2])
        self.up3 = Up(ch[2], ch[1], ch[1])
        self.up4 = Up(ch[1], ch[0], ch[0])
        self.head = nn.Conv2d(ch[0], 3, kernel_size=1)  # predicted RGB frame

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        b = self.down4(s3)

        B, C, H, W = b.shape
        seq = b.flatten(2).transpose(1, 2)
        seq = self.bottleneck_norm(seq + self.mamba(seq))
        b = seq.transpose(1, 2).reshape(B, C, H, W)

        x = self.up1(b, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        x = self.up4(x, s0)
        return self.head(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vadmamba_dataset", required=True, help="folder produced by vadmamba_prepare_data.py")
    ap.add_argument("--k", type=int, default=4, help="number of context frames used to predict the next frame")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_mamba_layers", type=int, default=4)
    ap.add_argument("--out_dir", default="runs/vadmamba")
    ap.add_argument("--num_workers", type=int, default=0,
                     help="DataLoader worker processes. Default 0 (safest on Windows -- num_workers>0 can hit "
                          "'Couldn't open shared file mapping' / error 1455 if the page file is small).")
    ap.add_argument("--resume", action="store_true", help="resume from out_dir/model_last.pt and out_dir/training_log.csv if they exist")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "model.pt")
    last_model_path = os.path.join(args.out_dir, "model_last.pt")
    log_path = os.path.join(args.out_dir, "training_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,epoch_minutes\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds = NormalRunsDataset(
        os.path.join(args.vadmamba_dataset, "images", "train"),
        os.path.join(args.vadmamba_dataset, "manifest_train.json"),
        k=args.k)
    print(f"Normal-run training windows (k={args.k}): {len(train_ds)}")
    if len(train_ds) == 0:
        raise RuntimeError(
            f"No training windows found -- need at least k+1={args.k + 1} consecutive normal frames in a "
            f"row somewhere in a train video. Try a smaller --k, or check manifest_train.json.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = MambaPredictor(k=args.k, n_mamba_layers=args.n_mamba_layers).to(device)

    start_epoch = 1
    best_loss = float("inf")
    if args.resume and os.path.exists(last_model_path):
        model.load_state_dict(torch.load(last_model_path, map_location=device))
        print(f"Resumed weights from {last_model_path}")
        if os.path.exists(log_path):
            with open(log_path) as f:
                rows = [line.strip().split(",") for line in f.readlines()[1:] if line.strip()]
            if rows:
                start_epoch = int(rows[-1][0]) + 1
                best_loss = min(float(r[1]) for r in rows)  # train_loss column
                print(f"Resuming at epoch {start_epoch}, best train_loss so far: {best_loss:.4f}")
    elif args.resume:
        print(f"--resume set but no checkpoint found at {last_model_path}, starting from scratch")

    l1 = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_train_batches = len(train_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        epoch_start = time.time()
        for batch_i, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device).float(), y.to(device).float()
            optimizer.zero_grad()
            pred = model(x)
            loss = l1(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            if batch_i % 50 == 0 or batch_i == n_train_batches:
                elapsed = time.time() - epoch_start
                rate = batch_i / elapsed
                eta = (n_train_batches - batch_i) / rate if rate > 0 else float("nan")
                print(f"  epoch {epoch:03d} train batch {batch_i}/{n_train_batches}  "
                      f"loss={loss.item():.4f}  {rate:.2f} batch/s  eta {eta/60:.1f} min", flush=True)
        train_loss /= max(1, len(train_ds))
        epoch_minutes = (time.time() - epoch_start) / 60
        print(f"epoch {epoch:03d}  train_loss={train_loss:.4f}  ({epoch_minutes:.1f} min total)")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{epoch_minutes:.2f}\n")

        torch.save(model.state_dict(), last_model_path)

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), model_path)
            print(f"  -> new best train_loss={best_loss:.4f}, saved to {model_path}")

    print(f"\nDone. Best train_loss: {best_loss:.4f}")
    print(f"Model: {model_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
