"""
dl3_train.py

Fine-tunes a torchvision DeepLabV3+ (ResNet-50 backbone, COCO-with-VOC-labels
pretrained weights) on the dl3_dataset/ produced by dl3_prepare_data.py.

Unlike ultralytics (`yolo segment train ...`), torchvision has no ready-made
training CLI, so this script is the equivalent of that command for this
model. Kept intentionally simple (plain train loop, no AMP/scheduler/vis
grids like train.py's CNN trainer) since this is a baseline for comparison,
not the project's main model.

Binary segmentation is done as 2-class classification (background=0,
object=1) with CrossEntropyLoss, tracking best checkpoint by val F1 using
the exact same TP/FP/FN accounting as train.py / yolo_evaluate.py.

Install once (same torchvision your Mask R-CNN baseline will use, so only
one extra install beyond what YOLO26 needed):
    pip install torch torchvision --break-system-packages

Usage:
    python dl3_train.py --dl3_dataset D:\\Minor_Project\\dl3_dataset ^
                         --epochs 40 --out_dir D:\\Minor_Project\\runs\\dl3
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
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

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
        y = (mask > 0).astype(np.int64)  # H,W with values {0,1}
        return torch.from_numpy(x), torch.from_numpy(y)


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
    ap.add_argument("--dl3_dataset", required=True, help="folder produced by dl3_prepare_data.py")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out_dir", default="runs/dl3")
    ap.add_argument("--num_workers", type=int, default=0,
                     help="DataLoader worker processes. Default 0 (safest on Windows -- num_workers>0 can hit "
                          "'Couldn't open shared file mapping' / error 1455 if the page file is small). "
                          "If you want to raise this, either increase Windows' page file size (System > Advanced "
                          "> Performance > Advanced > Virtual memory) or just leave it at 0 -- your GPU forward "
                          "pass is the bottleneck here, not CPU image decoding.")
    ap.add_argument("--no_pretrained", action="store_true", help="train from random init instead of COCO/VOC pretrained weights")
    ap.add_argument("--resume", action="store_true", help="resume from out_dir/model_last.pt and out_dir/training_log.csv if they exist (e.g. after a crash)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "model.pt")            # best-F1 checkpoint
    last_model_path = os.path.join(args.out_dir, "model_last.pt")   # most recent epoch, in case of crash
    log_path = os.path.join(args.out_dir, "training_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,precision,recall,f1,iou,epoch_minutes\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds = SegDataset(os.path.join(args.dl3_dataset, "images", "train"),
                           os.path.join(args.dl3_dataset, "masks", "train"))
    val_ds = SegDataset(os.path.join(args.dl3_dataset, "images", "val"),
                         os.path.join(args.dl3_dataset, "masks", "val"))
    print(f"Train frames: {len(train_ds)}  Val frames: {len(val_ds)}")

    # Object pixels are a small minority of each frame. Unweighted CrossEntropyLoss
    # lets the model reach a very low loss by predicting "background" everywhere --
    # that gives a great-looking train_loss curve while val precision/recall/F1 stay
    # at exactly 0.000 (0 true positives, 0 false positives -- the model just never
    # predicts the foreground class). Estimate class frequency from a sample of
    # train masks and weight the loss so missing the rare class actually costs
    # something.
    sample_paths = train_ds.image_paths[::max(1, len(train_ds) // 500)][:500]
    fg_px = total_px = 0
    for p in sample_paths:
        mpath = os.path.join(train_ds.masks_dir, os.path.basename(p))
        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            fg_px += int((m > 0).sum())
            total_px += m.size
    fg_frac = fg_px / total_px if total_px else 0.5
    fg_frac = min(max(fg_frac, 1e-4), 0.5)  # guard against div-by-zero / degenerate samples
    bg_weight = 1.0
    fg_weight = min((1 - fg_frac) / fg_frac, 50.0)  # cap so a near-empty sample doesn't blow up the loss
    print(f"Foreground pixel fraction (sampled from {len(sample_paths)} train masks): {fg_frac:.4f}  "
          f"-> class weights [bg={bg_weight:.2f}, fg={fg_weight:.2f}]")
    class_weights = torch.tensor([bg_weight, fg_weight], dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    weights = None if args.no_pretrained else DeepLabV3_ResNet50_Weights.DEFAULT
    # weights_backbone defaults to ImageNet-pretrained even when weights=None;
    # pass None explicitly too so --no_pretrained is genuinely from-scratch
    # (also what makes offline smoke-testing possible)
    model = deeplabv3_resnet50(weights=weights, weights_backbone=None if args.no_pretrained else "IMAGENET1K_V2",
                                num_classes=21 if weights else 2)
    if weights:
        # replace the (COCO/VOC, 21-class) classifier head with a fresh 2-class head;
        # everything below the head (backbone + ASPP) keeps the pretrained weights
        in_ch = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(in_ch, 2, kernel_size=1)
        if model.aux_classifier is not None:
            in_ch_aux = model.aux_classifier[4].in_channels
            model.aux_classifier[4] = nn.Conv2d(in_ch_aux, 2, kernel_size=1)
    model = model.to(device)

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
                best_f1 = max(float(r[4]) for r in rows)  # f1 column
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
            loss = ce(out["out"], y)
            if "aux" in out:
                loss = loss + 0.4 * ce(out["aux"], y)
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
                pred = model(x)["out"].argmax(dim=1).cpu().numpy()
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

        torch.save(model.state_dict(), last_model_path)  # always overwrite, so a crash only loses the current epoch

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save(model.state_dict(), model_path)
            print(f"  -> new best F1={best_f1:.3f}, saved to {model_path}")

    print(f"\nDone. Best val F1: {best_f1:.3f}")
    print(f"Model: {model_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
