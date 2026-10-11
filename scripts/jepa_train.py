"""
jepa_train.py

Fine-tunes a lightweight segmentation decoder on top of a FROZEN, pretrained
I-JEPA encoder (Meta's "Image-based Joint-Embedding Predictive
Architecture") on the jepa_dataset/ produced by jepa_prepare_data.py.

WHAT "USING JEPA" ACTUALLY MEANS HERE (read this before running):
JEPA is not a segmentation architecture -- it's a self-supervised
pretraining method. The public I-JEPA release is an encoder that was
trained to predict the embeddings of masked image regions from visible
ones (no pixel reconstruction, no labels). What Meta actually published as
downloadable weights is that encoder alone, e.g. `facebook/ijepa_vith16_1k`
(ViT-Huge/16, pretrained on ImageNet-1K) -- there is no official I-JEPA
segmentation head. So "using JEPA" for this comparison means: load that
pretrained encoder, FREEZE it (no gradients -- this is what makes it fit
in 6GB VRAM at all, since ViT-Huge is ~630M params), run your frames
through it to get patch embeddings, and train a small trainable decoder
on top that upsamples those embeddings back to a per-pixel mask. This is
the standard way JEPA-style encoders get evaluated on dense prediction
tasks (linear-probe / lightweight-decoder protocol), not a shortcut.

Also note: the object-detection literature's newest ideas (SAM2, DINOv3,
JEPA-based segmentation) are largely built this same way -- frozen
foundation encoder + light trainable head -- so this baseline is a
reasonably faithful "new architecture" comparison point, just not a
from-scratch trainable one like your CNN or the Mamba U-Net baseline.

Install (in addition to torch/torchvision, which you already have):
    pip install transformers

First run will download the encoder weights (~2.5GB, facebook/ijepa_vith16_1k)
from Hugging Face -- needs an internet connection once, then it's cached
locally (~/.cache/huggingface).

Usage:
    python jepa_train.py --jepa_dataset D:\\Minor_Project\\jepa_dataset ^
                          --epochs 40 --out_dir D:\\Minor_Project\\runs\\jepa
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
from transformers import IJepaModel

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ENCODER_NAME = "facebook/ijepa_vith16_1k"  # ViT-Huge/16, ImageNet-1K pretrained
PATCH_SIZE = 16


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


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class JepaDecoder(nn.Module):
    """Takes the frozen encoder's patch-token grid and upsamples it back to
    a per-pixel 2-class mask. patch_size=16 means 4 doublings (2^4=16)."""

    def __init__(self, in_ch, mid_ch=256):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, mid_ch, kernel_size=1)
        self.up1 = UpBlock(mid_ch, mid_ch // 2)
        self.up2 = UpBlock(mid_ch // 2, mid_ch // 4)
        self.up3 = UpBlock(mid_ch // 4, mid_ch // 8)
        self.up4 = UpBlock(mid_ch // 8, mid_ch // 8)
        self.head = nn.Conv2d(mid_ch // 8, 2, kernel_size=1)

    def forward(self, tokens_grid):
        x = self.proj(tokens_grid)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
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


def encode(encoder, x, patch_size, device):
    """Frozen forward pass through the I-JEPA encoder -> reshaped patch grid."""
    with torch.no_grad():
        out = encoder(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state  # B, N, C
    B, N, C = out.shape
    H = W = int(N ** 0.5)  # square input -> square patch grid
    return out.transpose(1, 2).reshape(B, C, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa_dataset", required=True, help="folder produced by jepa_prepare_data.py")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=4, help="kept small by default -- ViT-Huge forward pass is the memory bottleneck, not the decoder")
    ap.add_argument("--lr", type=float, default=1e-3, help="only the decoder trains, so this can be higher than a full fine-tune LR")
    ap.add_argument("--out_dir", default="runs/jepa")
    ap.add_argument("--num_workers", type=int, default=0,
                     help="DataLoader worker processes. Default 0 (safest on Windows -- num_workers>0 can hit "
                          "'Couldn't open shared file mapping' / error 1455 if the page file is small).")
    ap.add_argument("--resume", action="store_true", help="resume decoder weights from out_dir/model_last.pt and out_dir/training_log.csv if they exist")
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

    train_ds = SegDataset(os.path.join(args.jepa_dataset, "images", "train"),
                           os.path.join(args.jepa_dataset, "masks", "train"))
    val_ds = SegDataset(os.path.join(args.jepa_dataset, "images", "val"),
                         os.path.join(args.jepa_dataset, "masks", "val"))
    print(f"Train frames: {len(train_ds)}  Val frames: {len(val_ds)}")

    # same class-imbalance fix as every other baseline in this comparison
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

    print(f"Loading frozen encoder {ENCODER_NAME} (first run downloads ~2.5GB)...")
    encoder = IJepaModel.from_pretrained(ENCODER_NAME).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    hidden_size = encoder.config.hidden_size
    print(f"Encoder loaded, hidden_size={hidden_size}, frozen ({sum(p.numel() for p in encoder.parameters())/1e6:.0f}M params)")

    decoder = JepaDecoder(in_ch=hidden_size).to(device)

    start_epoch = 1
    best_f1 = -1.0
    if args.resume and os.path.exists(last_model_path):
        decoder.load_state_dict(torch.load(last_model_path, map_location=device))
        print(f"Resumed decoder weights from {last_model_path}")
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
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        decoder.train()
        train_loss = 0.0
        epoch_start = time.time()
        for batch_i, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device).float(), y.to(device)
            tokens_grid = encode(encoder, x, PATCH_SIZE, device)  # frozen, no_grad inside
            optimizer.zero_grad()
            out = decoder(tokens_grid)
            out = nn.functional.interpolate(out, size=y.shape[-2:], mode="bilinear", align_corners=False)
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

        decoder.eval()
        conf = ConfusionAccumulator()
        val_start = time.time()
        with torch.no_grad():
            for batch_i, (x, y) in enumerate(val_loader, 1):
                x = x.to(device).float()
                tokens_grid = encode(encoder, x, PATCH_SIZE, device)
                out = decoder(tokens_grid)
                out = nn.functional.interpolate(out, size=y.shape[-2:], mode="bilinear", align_corners=False)
                pred = out.argmax(dim=1).cpu().numpy()
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

        torch.save(decoder.state_dict(), last_model_path)

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save(decoder.state_dict(), model_path)
            print(f"  -> new best F1={best_f1:.3f}, saved to {model_path}")

    print(f"\nDone. Best val F1: {best_f1:.3f}")
    print(f"Model (decoder only -- encoder is the pretrained {ENCODER_NAME}, unchanged): {model_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
