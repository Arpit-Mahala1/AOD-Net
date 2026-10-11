"""
mamba_unet_train.py

Trains a Mamba U-Net baseline from scratch on the mamba_unet_dataset/
produced by mamba_unet_prepare_data.py.

ARCHITECTURE NOTE (read this before running):
"Mamba" here means a pure-PyTorch Selective State Space Model block from
the `mambapy` package -- NOT the official `mamba-ssm` package. The
official package needs a custom CUDA extension (causal-conv1d + a hand-written
selective-scan CUDA kernel) that's built from source; it's Linux-only in
practice and is a well-documented multi-hour-to-multi-day install fight on
Windows (search "mamba-ssm windows install" and you'll find dozens of
unresolved issues -- even people with matching CUDA/PyTorch versions hit
build failures). `mambapy` reimplements the same selective-scan math as a
parallel cumulative-sum operation in plain PyTorch, so it installs with a
single `pip install mambapy` on any platform (CPU, CUDA, Apple Silicon),
no compiler needed. It's slower per-step than the official kernel, but for
a comparison run at this scale (256x256 frames, ~16x16 bottleneck) that
difference is not the bottleneck -- your GPU forward/backward pass is.

Architecture: a standard 4-level U-Net (conv downsample / upsample with
skip connections, same shape as your own CNN's encoder-decoder), but the
bottleneck -- where a normal U-Net just applies a couple more conv layers --
is instead a stack of Mamba (selective SSM) blocks operating over the
flattened 16x16 spatial grid treated as a length-256 sequence. This follows
the same design idea as papers like VM-UNet / U-Mamba: convolutions handle
local texture, the SSM handles global/long-range context that a small
conv receptive field can't reach, replacing what a full self-attention
bottleneck would cost quadratically.

Trained from scratch (no pretrained Mamba-U-Net checkpoint exists for
natural images at this scale) -- unlike DeepLabV3+/Mask R-CNN/YOLO26, which
all start from a pretrained backbone. Worth stating plainly in your
write-up: this is architecture novelty vs. transfer learning, not an
apples-to-apples "does pretraining help" comparison with the other
baselines.

Install (in addition to torch/torchvision, which you already have):
    pip install mambapy

Usage:
    python mamba_unet_train.py --mamba_unet_dataset D:\\Minor_Project\\mamba_unet_dataset ^
                                --epochs 40 --out_dir D:\\Minor_Project\\runs\\mamba_unet
"""

import argparse
import glob
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


class SegDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        self.masks_dir = masks_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        ipath = self.image_paths[idx]
        name = os.path.basename(ipath)
        img = cv2.imread(ipath)
        mask_path = os.path.join(self.masks_dir, name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else np.zeros(img.shape[:2], np.uint8)
        x = normalize(img)
        y = (mask > 0).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


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


class MambaUNet(nn.Module):
    """4-level U-Net; bottleneck replaced with a stack of Mamba SSM blocks
    scanning the flattened spatial grid (see module docstring)."""

    def __init__(self, n_mamba_layers=4, d_state=16):
        super().__init__()
        ch = [32, 64, 128, 256, 512]  # stem, d1, d2, d3, bottleneck
        self.stem = ConvBlock(3, ch[0])
        self.down1 = Down(ch[0], ch[1])
        self.down2 = Down(ch[1], ch[2])
        self.down3 = Down(ch[2], ch[3])
        self.down4 = Down(ch[3], ch[4])  # -> 16x16 spatial at 256 input

        self.mamba = Mamba(MambaConfig(d_model=ch[4], n_layers=n_mamba_layers, d_state=d_state))
        self.bottleneck_norm = nn.LayerNorm(ch[4])

        self.up1 = Up(ch[4], ch[3], ch[3])
        self.up2 = Up(ch[3], ch[2], ch[2])
        self.up3 = Up(ch[2], ch[1], ch[1])
        self.up4 = Up(ch[1], ch[0], ch[0])
        self.head = nn.Conv2d(ch[0], 2, kernel_size=1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        b = self.down4(s3)  # B, C, H, W  (H=W=16 for 256x256 input)

        B, C, H, W = b.shape
        seq = b.flatten(2).transpose(1, 2)          # B, H*W, C  -- flatten spatial grid to a sequence
        seq = self.bottleneck_norm(seq + self.mamba(seq))  # residual Mamba block, pre-norm style
        b = seq.transpose(1, 2).reshape(B, C, H, W)  # back to spatial

        x = self.up1(b, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        x = self.up4(x, s0)
        return self.head(x)


class ConfusionAccumulator:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def update(self, preds_bin, targets_bin):
        preds_bin, targets_bin = preds_bin.astype(bool), targets_bin.astype(bool)
        self.tp += int(np.logical_and(preds_bin, targets_bin).sum())
        self.fp += int(np.logical_and(preds_bin, ~targets_bin).sum())
        self.fn += int(np.logical_and(~preds_bin, targets_bin).sum())

    def metrics(self, eps=1e-7):
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mamba_unet_dataset", required=True, help="folder produced by mamba_unet_prepare_data.py")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_mamba_layers", type=int, default=4)
    ap.add_argument("--out_dir", default="runs/mamba_unet")
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
            f.write("epoch,train_loss,precision,recall,f1,iou,epoch_minutes\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds = SegDataset(os.path.join(args.mamba_unet_dataset, "images", "train"),
                           os.path.join(args.mamba_unet_dataset, "masks", "train"))
    val_ds = SegDataset(os.path.join(args.mamba_unet_dataset, "images", "val"),
                         os.path.join(args.mamba_unet_dataset, "masks", "val"))
    print(f"Train frames: {len(train_ds)}  Val frames: {len(val_ds)}")

    # same class-imbalance fix as dl3_train.py: object pixels are a small
    # minority of each frame, so unweighted CrossEntropyLoss collapses to
    # predicting "background" everywhere (great train_loss, val recall stuck
    # at exactly 0.000). Estimate class frequency from a sample of train
    # masks and weight the loss accordingly.
    sample_paths = train_ds.image_paths[::max(1, len(train_ds) // 500)][:500]
    fg_px = total_px = 0
    for p in sample_paths:
        mpath = os.path.join(train_ds.masks_dir, os.path.basename(p))
        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            fg_px += int((m > 0).sum())
            total_px += m.size
    fg_frac = fg_px / total_px if total_px else 0.5
    fg_frac = min(max(fg_frac, 1e-4), 0.5)
    bg_weight = 1.0
    fg_weight = min((1 - fg_frac) / fg_frac, 50.0)
    print(f"Foreground pixel fraction (sampled from {len(sample_paths)} train masks): {fg_frac:.4f}  "
          f"-> class weights [bg={bg_weight:.2f}, fg={fg_weight:.2f}]")
    class_weights = torch.tensor([bg_weight, fg_weight], dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MambaUNet(n_mamba_layers=args.n_mamba_layers).to(device)

    start_epoch = 1
    best_f1 = -1.0
    if args.resume and os.path.exists(last_model_path):
        model.load_state_dict(torch.load(last_model_path, map_location=device))
        print(f"Resumed weights from {last_model_path}")
        if os.path.exists(log_path):
            with open(log_path) as f:
                rows = [line.strip().split(",") for line in f.readlines()[1:] if line.strip()]
            if rows:
                start_epoch = int(rows[-1][0]) + 1
                best_f1 = max(float(r[4]) for r in rows)  # f1 column (epoch,train_loss,precision,recall,f1,iou,epoch_minutes)
                print(f"Resuming at epoch {start_epoch}, best F1 so far: {best_f1:.3f}")
    elif args.resume:
        print(f"--resume set but no checkpoint found at {last_model_path}, starting from scratch")

    ce = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        epoch_start = time.time()
        for batch_i, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device).float(), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = ce(out, y)
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

        model.eval()
        conf = ConfusionAccumulator()
        val_start = time.time()
        with torch.no_grad():
            for batch_i, (x, y) in enumerate(val_loader, 1):
                x = x.to(device).float()
                pred = model(x).argmax(dim=1).cpu().numpy()
                conf.update(pred > 0, y.numpy() > 0)
                if batch_i % 50 == 0 or batch_i == n_val_batches:
                    print(f"  epoch {epoch:03d} val batch {batch_i}/{n_val_batches}  "
                          f"({time.time() - val_start:.1f}s elapsed)", flush=True)
        m = conf.metrics()
        epoch_minutes = (time.time() - epoch_start) / 60
        print(f"epoch {epoch:03d}  train_loss={train_loss:.4f}  "
              f"precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}  iou={m['iou']:.3f}  "
              f"({epoch_minutes:.1f} min total)")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{m['precision']:.6f},{m['recall']:.6f},{m['f1']:.6f},{m['iou']:.6f},{epoch_minutes:.2f}\n")

        torch.save(model.state_dict(), last_model_path)

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save(model.state_dict(), model_path)
            print(f"  -> new best F1={best_f1:.3f}, saved to {model_path}")

    print(f"\nDone. Best val F1: {best_f1:.3f}")
    print(f"Model: {model_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
