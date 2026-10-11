"""
train.py

Trains a small U-Net (pretrained ResNet18 encoder) to segment "static foreign
object" pixels, from the (frame, background) pairs produced by dataset_prep.py.

This is a per-frame model -- it does NOT know about P/C. That's applied
afterwards in infer.py. Keeping the model's job simple (visual segmentation
only) is what makes this trainable with a small dataset.

Sized for RTX 4050 6GB: 256x256 input, resnet18 encoder, batch size 8, AMP.

Every epoch prints train/val loss plus pixel-level precision/recall/F1/IoU on
the "object" class (accuracy alone is close to meaningless here since most
pixels are background). A CSV log and a loss/metric curve PNG are written to
--out_dir each epoch, and every --vis_every epochs a grid of a few fixed
validation samples (frame / ground truth / prediction) is saved so you can
watch it visually improve instead of just reading numbers.

Install once:
    pip install torch torchvision segmentation-models-pytorch matplotlib --break-system-packages

Usage:
    python train.py --data_root D:\\Minor_Project\\training_data --epochs 40 --out_dir D:\\Minor_Project\\runs\\run1
"""

import argparse
import csv
import glob
import os
import random
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

import segmentation_models_pytorch as smp

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)  # C,H,W


def denormalize_to_rgb(x_chw):
    """Inverse of normalize(), for the first 3 channels only (the frame)."""
    img = x_chw.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def augment_pair(frame, bg, label):
    """
    Applied only to training samples. Goal: stop the model from memorizing
    exact pixel patterns of the ~21 training backgrounds (which is what was
    causing overfitting within 1-2 epochs) and instead learn the general
    "how does a static foreign object differ from its background" relationship.
    """
    h, w = frame.shape[:2]

    if random.random() < 0.5:  # horizontal flip
        frame = cv2.flip(frame, 1)
        bg = cv2.flip(bg, 1)
        label = cv2.flip(label, 1)

    if random.random() < 0.5:  # brightness/contrast jitter, applied identically to frame+bg
        alpha = 1.0 + random.uniform(-0.2, 0.2)  # contrast
        beta = random.uniform(-20, 20)            # brightness
        frame = np.clip(frame.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        bg = np.clip(bg.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if random.random() < 0.5:  # random crop-and-zoom
        scale = random.uniform(0.85, 1.0)
        ch, cw = max(1, int(h * scale)), max(1, int(w * scale))
        y0 = random.randint(0, h - ch)
        x0 = random.randint(0, w - cw)
        frame = cv2.resize(frame[y0:y0 + ch, x0:x0 + cw], (w, h))
        bg = cv2.resize(bg[y0:y0 + ch, x0:x0 + cw], (w, h))
        label = cv2.resize(label[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_NEAREST)

    return frame, bg, label


class FrameDataset(Dataset):
    def __init__(self, data_root, augment=False):
        self.frame_paths = sorted(glob.glob(os.path.join(data_root, "frames", "*.png")))
        self.bg_dir = os.path.join(data_root, "backgrounds")
        self.label_dir = os.path.join(data_root, "labels")
        self.augment = augment

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        fpath = self.frame_paths[idx]
        name = os.path.basename(fpath)
        frame = cv2.imread(fpath)
        bg = cv2.imread(os.path.join(self.bg_dir, name))
        label = cv2.imread(os.path.join(self.label_dir, name), cv2.IMREAD_GRAYSCALE)

        if self.augment:
            frame, bg, label = augment_pair(frame, bg, label)

        frame_t = normalize(frame)
        bg_t = normalize(bg)
        x = np.concatenate([frame_t, bg_t], axis=0)  # 6,H,W
        y = (label > 0).astype(np.float32)[None, :, :]  # 1,H,W

        return torch.from_numpy(x), torch.from_numpy(y)


def set_seed(seed):
    """
    Seeds Python's random, numpy, and torch (CPU + CUDA) so that model weight
    initialization and training-batch shuffling are reproducible across runs.
    Without this, the video/frame SPLIT was already reproducible (it has its
    own --split_seed), but the model's random decoder init and DataLoader
    shuffling were not -- meaning two runs with identical flags could land in
    different local minima purely from that randomness, making any A/B
    comparison between hyperparameters (e.g. encoder_depth 4 vs 5) unreliable
    unless both runs use the same --seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch.manual_seed alone does NOT make CUDA convolutions reproducible --
    # cuDNN's default "benchmark" mode picks algorithms non-deterministically
    # for speed. Without this, two runs with an identical --seed can still
    # diverge on GPU, which makes any A/B comparison (e.g. testing --grad_clip
    # values) unreliable. This costs some training speed in exchange for
    # runs actually being comparable.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def group_split_indices(frame_paths, val_fraction=0.2, seed=42):
    """
    Splits by SAMPLE FOLDER (whole video), not by individual frame. Random
    frame-level splitting leaks information because consecutive/nearby frames
    from the same video are near-duplicates (same background, same object,
    barely-moved camera noise) -- a model can partly memorize a video's own
    appearance rather than learning to generalize, which inflates val metrics
    without meaning much for the actual unseen test videos.

    Filenames are "{sample_name}_{frame_idx:06d}.png" -- group by everything
    before the last underscore. Returns index lists, not Subset objects, so
    the caller can apply augmentation to the train indices only.
    """
    groups = defaultdict(list)
    for idx, fpath in enumerate(frame_paths):
        name = os.path.splitext(os.path.basename(fpath))[0]
        sample_name = name.rsplit("_", 1)[0]
        groups[sample_name].append(idx)

    sample_names = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(sample_names)

    n_val = max(1, round(len(sample_names) * val_fraction))
    val_samples = sorted(sample_names[:n_val])
    train_samples = sorted(sample_names[n_val:])

    train_idx = [i for s in train_samples for i in groups[s]]
    val_idx = [i for s in val_samples for i in groups[s]]

    return train_idx, val_idx, train_samples, val_samples


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


class ConfusionAccumulator:
    """Accumulates TP/FP/FN/TN across batches, then derives metrics once."""

    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, preds_bin, targets_bin):
        preds_bin = preds_bin.astype(bool)
        targets_bin = targets_bin.astype(bool)
        self.tp += np.logical_and(preds_bin, targets_bin).sum()
        self.fp += np.logical_and(preds_bin, ~targets_bin).sum()
        self.fn += np.logical_and(~preds_bin, targets_bin).sum()
        self.tn += np.logical_and(~preds_bin, ~targets_bin).sum()

    def metrics(self, eps=1e-7):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "iou": iou}


def save_visual_samples(model, device, val_ds, sample_indices, epoch, out_dir, row_labels=None, prob_thresh=0.5):
    """Saves a grid: rows = fixed validation samples, cols = frame / ground truth / prediction."""
    model.eval()
    n = len(sample_indices)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for row, idx in enumerate(sample_indices):
            x, y = val_ds[idx]
            x_in = x.unsqueeze(0).to(device)
            logits = model(x_in)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred_bin = (prob > prob_thresh).astype(np.uint8) * 255

            frame_rgb = denormalize_to_rgb(x.numpy()[:3])
            gt = (y.numpy()[0] * 255).astype(np.uint8)
            has_object = gt.max() > 0

            axes[row, 0].imshow(frame_rgb)
            axes[row, 0].set_title("frame" if row == 0 else "")
            axes[row, 1].imshow(gt, cmap="gray", vmin=0, vmax=255)
            axes[row, 1].set_title("ground truth" if row == 0 else "")
            axes[row, 2].imshow(pred_bin, cmap="gray", vmin=0, vmax=255)
            axes[row, 2].set_title("prediction" if row == 0 else "")
            row_tag = f"[{'object' if has_object else 'empty'}]"
            if row_labels:
                row_tag = f"{row_labels[row]} {row_tag}"
            axes[row, 0].set_ylabel(row_tag, fontsize=8, rotation=0, ha="right", va="center")
            axes[row, 0].axis("on")
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])
            for spine in axes[row, 0].spines.values():
                spine.set_visible(False)
            for c in (1, 2):
                axes[row, c].axis("off")

    fig.suptitle(f"epoch {epoch}")
    fig.tight_layout()
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    fig.savefig(os.path.join(out_dir, "samples", f"epoch_{epoch:03d}.png"), dpi=100)
    plt.close(fig)


def save_curves(log_rows, out_dir):
    epochs = [r["epoch"] for r in log_rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [r["train_loss"] for r in log_rows], label="train_loss")
    axes[0].plot(epochs, [r["val_loss"] for r in log_rows], label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [r["precision"] for r in log_rows], label="precision")
    axes[1].plot(epochs, [r["recall"] for r in log_rows], label="recall")
    axes[1].plot(epochs, [r["f1"] for r in log_rows], label="f1")
    axes[1].plot(epochs, [r["iou"] for r in log_rows], label="iou")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Validation metrics (object class)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=110)
    plt.close(fig)


def pick_visual_indices(dataset, val_idx, n_positive=2, n_negative=2, seed=0):
    """
    val_idx walks through held-out videos start-to-end, so the first few
    positions are almost always early, pre-appearance frames of the first
    video -- always black, uninformative. Instead, shuffle and grab a mix of
    frames that DO and DON'T contain an object, so the saved grid actually
    shows whether the model can find the thing.
    """
    order = list(enumerate(val_idx))
    random.Random(seed).shuffle(order)
    positives, negatives = [], []
    for pos, global_idx in order:
        name = os.path.basename(dataset.frame_paths[global_idx])
        label = cv2.imread(os.path.join(dataset.label_dir, name), cv2.IMREAD_GRAYSCALE)
        has_object = label is not None and label.max() > 0
        if has_object and len(positives) < n_positive:
            positives.append(pos)
        elif not has_object and len(negatives) < n_negative:
            negatives.append(pos)
        if len(positives) >= n_positive and len(negatives) >= n_negative:
            break
    return positives + negatives


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out_dir", default="runs/run1", help="where model, logs, curves, and sample images go")
    ap.add_argument("--vis_every", type=int, default=5, help="save a visual sample grid every N epochs")
    ap.add_argument("--num_vis_samples", type=int, default=4)
    ap.add_argument("--val_fraction", type=float, default=0.2, help="fraction of VIDEOS (not frames) held out for validation")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--encoder_lr_mult", type=float, default=0.1,
                     help="pretrained encoder trains at lr * this; decoder trains at full lr")
    ap.add_argument("--patience", type=int, default=8,
                     help="stop early if val F1 doesn't improve for this many epochs")
    ap.add_argument("--encoder_depth", type=int, default=5, choices=[3, 4, 5],
                     help="fewer stages = less downsampling = small objects survive to the deepest layer better, "
                          "at the cost of less semantic context. Try 4 if small objects are being missed entirely.")
    ap.add_argument("--pos_weight", type=float, default=1.0,
                     help="upweights the object class in BCE loss (>1 pushes the model to call more pixels "
                          "positive -- helps a 'confident tiny point' expand into the full object silhouette)")
    ap.add_argument("--seed", type=int, default=42,
                     help="seeds model weight init and training-batch shuffling. Use the SAME --seed when "
                          "A/B comparing hyperparameters (e.g. encoder_depth) -- otherwise differences between "
                          "runs may just be random init luck, not a real effect of the flag you changed.")
    ap.add_argument("--grad_clip", type=float, default=0.0,
                     help="caps the gradient norm to this value each step, regardless of loss magnitude. "
                          "0 = disabled (default) -- the gradient norm is still measured and printed/logged every "
                          "epoch either way, so you can look at real numbers before picking a threshold instead "
                          "of guessing one.")
    ap.add_argument("--architecture", choices=["unet", "deeplabv3plus"], default="unet",
                     help="unet (default, current pipeline) or deeplabv3plus, which adds ASPP (dilated "
                          "convolutions) for multi-scale context WITHOUT reducing encoder_depth -- an alternative "
                          "way to address small-object sensitivity that doesn't touch the downsampling depth "
                          "encoder_depth=4 has repeatedly been unstable at on this dataset. Recommended: leave "
                          "--encoder_depth at its default (5) when using deeplabv3plus -- that combination is "
                          "untested and unsupported for other depths.")
    ap.add_argument("--disable_amp", action="store_true",
                     help="run in full fp32 instead of mixed precision. Use this if grad_norm shows inf/nan in "
                          "the epoch log -- fp16 can silently overflow/underflow on extreme predictions (very "
                          "confident, near-0 or near-1) that a stable fp32 forward pass wouldn't. Slower and uses "
                          "more memory, but removes that entire class of numerical failure.")
    args = ap.parse_args()

    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "model.pt")
    log_path = os.path.join(args.out_dir, "metrics_log.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}  |  architecture={args.architecture}  encoder_depth={args.encoder_depth}")

    dataset_for_indices = FrameDataset(args.data_root, augment=False)
    print(f"Dataset size: {len(dataset_for_indices)} frames")
    train_idx, val_idx, train_samples, val_samples = group_split_indices(
        dataset_for_indices.frame_paths, val_fraction=args.val_fraction, seed=args.split_seed)
    print(f"Train: {len(train_samples)} videos, {len(train_idx)} frames -> {train_samples}")
    print(f"Val (held out, ENTIRE videos never seen in training): {len(val_samples)} videos, {len(val_idx)} frames -> {val_samples}")

    train_dataset_aug = FrameDataset(args.data_root, augment=True)
    val_dataset_noaug = dataset_for_indices  # augment=False
    train_ds = Subset(train_dataset_aug, train_idx)
    val_ds = Subset(val_dataset_noaug, val_idx)

    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
                               generator=train_loader_generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    vis_indices = pick_visual_indices(dataset_for_indices, val_idx,
                                       n_positive=max(1, args.num_vis_samples // 2),
                                       n_negative=max(1, args.num_vis_samples - args.num_vis_samples // 2))
    vis_labels = [os.path.basename(dataset_for_indices.frame_paths[val_idx[i]]) for i in vis_indices]
    print(f"Visual sample frames: {vis_labels}")

    decoder_channels_by_depth = {5: (256, 128, 64, 32, 16), 4: (256, 128, 64, 32), 3: (256, 128, 64)}
    if args.architecture == "unet":
        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            encoder_depth=args.encoder_depth,
            decoder_channels=decoder_channels_by_depth[args.encoder_depth],
            in_channels=6,   # frame RGB + background RGB stacked
            classes=1,
        ).to(device)
    else:  # deeplabv3plus
        model = smp.DeepLabV3Plus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            encoder_depth=args.encoder_depth,
            in_channels=6,   # frame RGB + background RGB stacked
            classes=1,
        ).to(device)

    pos_weight_t = torch.tensor([args.pos_weight], device=device) if args.pos_weight != 1.0 else None
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    # Differential learning rate: the pretrained encoder already knows good
    # general features, so it trains slower/gentler than the randomly-init
    # decoder. This, plus weight decay, is what keeps the encoder from being
    # yanked into overfitting the ~21 training scenes within 1-2 epochs.
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": args.lr * args.encoder_lr_mult},
        {"params": decoder_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not args.disable_amp))

    best_f1 = -1.0
    epochs_since_improvement = 0
    log_rows = []

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "accuracy", "precision", "recall", "f1", "iou",
                                 "grad_norm_mean", "grad_norm_max"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        grad_norms = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda" and not args.disable_amp)):
                logits = model(x)
                loss = bce(logits, y) + dice_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # unscale AMP's fp16 gradients so norms/clipping below are meaningful
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=(args.grad_clip if args.grad_clip > 0 else float("inf")))
            grad_norms.append(float(total_norm))
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)
        mean_grad_norm = sum(grad_norms) / len(grad_norms)
        max_grad_norm = max(grad_norms)

        model.eval()
        val_loss = 0.0
        conf = ConfusionAccumulator()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=(device == "cuda" and not args.disable_amp)):
                    logits = model(x)
                    loss = bce(logits, y) + dice_loss(logits, y)
                val_loss += loss.item() * x.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                preds_bin = probs > 0.5
                targets_bin = y.cpu().numpy() > 0.5
                conf.update(preds_bin, targets_bin)
        val_loss /= len(val_ds)
        m = conf.metrics()

        print(f"epoch {epoch:03d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}  "
              f"iou={m['iou']:.3f}  acc={m['accuracy']:.3f}  "
              f"grad_norm(mean/max)={mean_grad_norm:.2f}/{max_grad_norm:.2f}")

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **m}
        log_rows.append(row)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, m["accuracy"], m["precision"], m["recall"],
                                     m["f1"], m["iou"], mean_grad_norm, max_grad_norm])

        save_curves(log_rows, args.out_dir)

        if epoch % args.vis_every == 0 or epoch == 1 or epoch == args.epochs:
            save_visual_samples(model, device, val_ds, vis_indices, epoch, args.out_dir, row_labels=vis_labels)

        # Track best by F1 on the object class, not loss -- with this much
        # class imbalance, loss can keep dropping slightly from getting the
        # (easy) background pixels more confidently right, without the
        # (hard, rare) object pixels actually improving. F1 tracks what you
        # actually care about.
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            epochs_since_improvement = 0
            torch.save(model.state_dict(), model_path)
            print(f"  -> new best F1={best_f1:.3f}, saved to {model_path}")
        else:
            epochs_since_improvement += 1

        scheduler.step(m["f1"])
        current_lrs = [g["lr"] for g in optimizer.param_groups]
        print(f"  lr(encoder,decoder)={current_lrs}  epochs_since_improvement={epochs_since_improvement}")

        if epochs_since_improvement >= args.patience:
            print(f"\nStopping early: no val F1 improvement for {args.patience} epochs. "
                  f"Best model (F1={best_f1:.3f}) is already saved at {model_path}.")
            break

    print(f"\nDone. Best val F1: {best_f1:.3f}")
    print(f"Model: {model_path}")
    print(f"Metrics log: {log_path}")
    print(f"Curves: {os.path.join(args.out_dir, 'training_curves.png')}")
    print(f"Sample grids: {os.path.join(args.out_dir, 'samples')}")


if __name__ == "__main__":
    main()
