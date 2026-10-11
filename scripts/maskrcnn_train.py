"""
maskrcnn_train.py

Fine-tunes a torchvision Mask R-CNN (ResNet-50-FPN, COCO-pretrained) on the
maskrcnn_dataset/ produced by maskrcnn_prepare_data.py.

Like DeepLabV3+, torchvision has no ready-made training CLI (unlike
ultralytics' `yolo segment train ...`), so this script is that command's
equivalent for this model.

Mask R-CNN is an INSTANCE segmentation model -- it needs a separate
(box, mask) pair per object, not one flat binary mask. Since
dataset_prep.py's raw label PNG is a union of every object present at that
moment, per-instance masks are recovered here via connected-component
analysis on each label PNG at load time (cv2.connectedComponents). This is
a reasonable approximation for this dataset (objects are placed apart, not
overlapping/touching), but note it: two touching objects in one frame would
be treated as a single instance, same caveat dataset_prep.py's own
extract_object_tracks() documents about mask-level (not per-frame) fusion.

Frames with zero instances (no object present yet) are valid Mask R-CNN
training examples (background-only) and are kept, not skipped.

Install once (torchvision only -- same install as the DeepLabV3+ baseline):
    pip install torch torchvision --break-system-packages

Usage:
    python maskrcnn_train.py --maskrcnn_dataset D:\\Minor_Project\\maskrcnn_dataset ^
                              --epochs 40 --out_dir D:\\Minor_Project\\runs\\maskrcnn
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

MIN_INSTANCE_AREA = 20  # px, at training_data's own resolution -- discard tiny connected-component noise


def extract_instances(mask):
    """mask: HxW uint8, 0/255 union label. Returns (boxes Nx4, masks NxHxW bool)."""
    n_labels, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    boxes, masks = [], []
    for lbl in range(1, n_labels):  # 0 is background
        inst = labels == lbl
        if inst.sum() < MIN_INSTANCE_AREA:
            continue
        ys, xs = np.where(inst)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        masks.append(inst)
    return np.array(boxes, dtype=np.float32), np.array(masks, dtype=np.uint8)


class InstanceDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        self.masks_dir = masks_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        ipath = self.image_paths[idx]
        name = os.path.basename(ipath)
        img = cv2.cvtColor(cv2.imread(ipath), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask_path = os.path.join(self.masks_dir, name)
        label_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else np.zeros(img.shape[:2], np.uint8)

        boxes, masks = extract_instances(label_mask)
        n = len(boxes)
        target = {
            "boxes": torch.as_tensor(boxes if n else np.zeros((0, 4), np.float32), dtype=torch.float32),
            "labels": torch.ones((n,), dtype=torch.int64),  # single foreground class = "object"
            "masks": torch.as_tensor(masks if n else np.zeros((0,) + img.shape[:2], np.uint8), dtype=torch.uint8),
            "image_id": torch.tensor([idx]),
        }
        x = torch.from_numpy(img.transpose(2, 0, 1))
        return x, target


def collate_fn(batch):
    return tuple(zip(*batch))


def pixel_metrics_from_masks(gt_union, pred_union):
    gt_pos, pred_pos = gt_union > 0, pred_union > 0
    tp = int(np.logical_and(gt_pos, pred_pos).sum())
    fp = int(np.logical_and(~gt_pos, pred_pos).sum())
    fn = int(np.logical_and(gt_pos, ~pred_pos).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return tp, fp, fn, precision, recall, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maskrcnn_dataset", required=True, help="folder produced by maskrcnn_prepare_data.py")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out_dir", default="runs/maskrcnn")
    ap.add_argument("--num_workers", type=int, default=0,
                     help="DataLoader worker processes. Default 0 (safest on Windows -- num_workers>0 can hit "
                          "'Couldn't open shared file mapping' / error 1455 if the page file is small).")
    ap.add_argument("--score_thresh", type=float, default=0.3, help="confidence threshold used for val F1 tracking during training")
    ap.add_argument("--no_pretrained", action="store_true", help="train from random init instead of COCO pretrained weights")
    ap.add_argument("--resume", action="store_true", help="resume from out_dir/model_last.pt and out_dir/training_log.csv if they exist (e.g. after a crash)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "model.pt")            # best-F1 checkpoint
    last_model_path = os.path.join(args.out_dir, "model_last.pt")   # most recent epoch, in case of crash
    log_path = os.path.join(args.out_dir, "training_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,precision,recall,f1,epoch_minutes\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds = InstanceDataset(os.path.join(args.maskrcnn_dataset, "images", "train"),
                                os.path.join(args.maskrcnn_dataset, "masks", "train"))
    val_ds = InstanceDataset(os.path.join(args.maskrcnn_dataset, "images", "val"),
                              os.path.join(args.maskrcnn_dataset, "masks", "val"))
    print(f"Train frames: {len(train_ds)}  Val frames: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    weights = None if args.no_pretrained else MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights,
                                   weights_backbone=None if args.no_pretrained else "IMAGENET1K_V2")
    # replace the (COCO, 91-class) box + mask heads with fresh 2-class
    # (background + object) heads; the backbone/FPN keep pretrained weights
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes=2)
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
                best_f1 = max(float(r[4]) for r in rows)  # f1 column (epoch,train_loss,precision,recall,f1,epoch_minutes)
                print(f"Resuming at epoch {start_epoch}, best F1 so far: {best_f1:.3f}")
    elif args.resume:
        print(f"--resume set but no checkpoint found at {last_model_path}, starting from scratch")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        epoch_start = time.time()
        for batch_i, (images, targets) in enumerate(train_loader, 1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            n_batches += 1
            if batch_i % 50 == 0 or batch_i == n_train_batches:
                elapsed = time.time() - epoch_start
                rate = batch_i / elapsed
                eta = (n_train_batches - batch_i) / rate if rate > 0 else float("nan")
                print(f"  epoch {epoch:03d} train batch {batch_i}/{n_train_batches}  "
                      f"loss={loss.item():.4f}  {rate:.2f} batch/s  eta {eta/60:.1f} min", flush=True)
        train_loss /= max(1, n_batches)

        model.eval()
        tot_tp = tot_fp = tot_fn = 0
        val_start = time.time()
        with torch.no_grad():
            for batch_i, (images, targets) in enumerate(val_loader, 1):
                h, w = images[0].shape[-2:]
                gt_union = np.zeros((h, w), dtype=np.uint8)
                for m in targets[0]["masks"].numpy():
                    gt_union = np.maximum(gt_union, m * 255)

                pred = model([images[0].to(device)])[0]
                pred_union = np.zeros((h, w), dtype=np.uint8)
                for score, mask in zip(pred["scores"].cpu().numpy(), pred["masks"].cpu().numpy()):
                    if score >= args.score_thresh:
                        pred_union = np.maximum(pred_union, (mask[0] > 0.5).astype(np.uint8) * 255)

                tp, fp, fn, _, _, _ = pixel_metrics_from_masks(gt_union, pred_union)
                tot_tp += tp
                tot_fp += fp
                tot_fn += fn
                if batch_i % 50 == 0 or batch_i == n_val_batches:
                    print(f"  epoch {epoch:03d} val batch {batch_i}/{n_val_batches}  "
                          f"({time.time() - val_start:.1f}s elapsed)", flush=True)

        precision = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        recall = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        epoch_minutes = (time.time() - epoch_start) / 60
        print(f"epoch {epoch:03d}  train_loss={train_loss:.4f}  "
              f"val precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}  ({epoch_minutes:.1f} min total)")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{precision:.6f},{recall:.6f},{f1:.6f},{epoch_minutes:.2f}\n")

        torch.save(model.state_dict(), last_model_path)  # always overwrite, so a crash only loses the current epoch

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), model_path)
            print(f"  -> new best F1={best_f1:.3f}, saved to {model_path}")

    print(f"\nDone. Best val F1: {best_f1:.3f}")
    print(f"Model: {model_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
