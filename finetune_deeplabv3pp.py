import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import models

from dataset_ORFD import ORFDFinetuneDataset


# ================== CONFIG ==================
ROUND_NUM = 2  # 0(baseline), 1, 2, ...

# CSVs (must have at least 'image_path' and 'gt_path' columns)
TRAIN_CSV = f"Unlabeled/finetuning_pool_round{ROUND_NUM}.csv"
VAL_CSV   = "OFRD/validation_ORFD.csv"   # change if different

IMG_SIZE   = (352, 640)   # (H, W) – must match dataset resize
BATCH_SIZE = 4
EPOCHS     = 15
LR         = 1e-4
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 2          # background + drivable
DRIVABLE_CLASS = 1       # index for drivable class

# Starting checkpoint (BDD-trained DeepLabV3+)
START_CKPT = "models/deeplabv3++/deeplabv3_drivable_best.pth"
# START_CKPT = f"models/deeplabv3++/deeplabv3_orfd_round{ROUND_NUM-1}.pth"


# Output checkpoints for this ORFD round
OUT_DIR = Path("models/deeplabv3++")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# LAST_CKPT = OUT_DIR / f"deeplabv3_orfd_round{ROUND_NUM}.pth"
BEST_CKPT =  OUT_DIR / f"deeplabv3_orfd_round{ROUND_NUM}.pth"
# ===========================================


def build_loaders():
    train_dataset = ORFDFinetuneDataset(
        csv_path=TRAIN_CSV,
        size=IMG_SIZE,
    )
    val_dataset = ORFDFinetuneDataset(
        csv_path=VAL_CSV,
        size=IMG_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    return train_loader, val_loader


def build_deeplabv3_mnv3(num_classes: int) -> nn.Module:
    """
    DeepLabV3+ with MobileNet V3 backbone from torchvision.
    """
    try:
        model = models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=None,
            num_classes=num_classes,
        )
    except TypeError:
        # older torchvision API
        model = models.segmentation.deeplabv3_mobilenet_v3_large(
            pretrained=False,
            num_classes=num_classes,
        )
    return model


def build_model():
    model = build_deeplabv3_mnv3(NUM_CLASSES)

    ckpt_path = Path(START_CKPT)
    if ckpt_path.is_file():
        print(f"[INFO] Loading starting weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys when loading START_CKPT ({len(missing)}):")
            print(missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            print(f"[WARN] Unexpected keys when loading START_CKPT ({len(unexpected)}):")
            print(unexpected[:10], "..." if len(unexpected) > 10 else "")
    else:
        print(f"[WARN] START_CKPT not found, training from scratch: {START_CKPT}")

    return model.to(DEVICE)


def calc_iou_binary(pred, target, eps=1e-6):
    """
    pred, target: (B,H,W) with {0,1}
    """
    pred = pred.view(-1).float()
    target = target.view(-1).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    if union <= 0:
        return torch.tensor(1.0, device=pred.device)
    return (inter + eps) / (union + eps)


def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    n_samples = 0

    criterion = nn.CrossEntropyLoss()  # logits (B,2,H,W) vs labels (B,H,W)

    for imgs, masks in tqdm(loader, desc=f"Epoch {epoch} [train]", ncols=100):
        imgs = imgs.to(DEVICE)            # (B,3,H,W)
        masks = masks.to(DEVICE)          # (B,1,H,W) with {0,1}
        labels = masks.squeeze(1).long()  # (B,H,W)

        optimizer.zero_grad()

        outputs = model(imgs)
        logits = outputs["out"]          # (B,2,h',w')

        # upsample logits to match label size
        logits_up = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = criterion(logits_up, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        with torch.no_grad():
            preds = torch.argmax(logits_up, dim=1)   # (B,H,W)
            pred_bin = (preds == DRIVABLE_CLASS).float()
            gt_bin = labels.float()

            iou = calc_iou_binary(pred_bin, gt_bin)
            running_iou += iou.item() * imgs.size(0)
            n_samples += imgs.size(0)

    avg_loss = running_loss / max(n_samples, 1)
    avg_iou  = running_iou / max(n_samples, 1)
    return avg_loss, avg_iou


def validate(model, loader, epoch):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    n_samples = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=f"Epoch {epoch} [val]", ncols=100):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            labels = masks.squeeze(1).long()

            outputs = model(imgs)
            logits = outputs["out"]

            logits_up = F.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss = criterion(logits_up, labels)
            running_loss += loss.item() * imgs.size(0)

            preds = torch.argmax(logits_up, dim=1)
            pred_bin = (preds == DRIVABLE_CLASS).float()
            gt_bin = labels.float()

            iou = calc_iou_binary(pred_bin, gt_bin)
            running_iou += iou.item() * imgs.size(0)
            n_samples += imgs.size(0)

    avg_loss = running_loss / max(n_samples, 1)
    avg_iou  = running_iou / max(n_samples, 1)
    return avg_loss, avg_iou


def main():
    train_loader, val_loader = build_loaders()
    model = build_model()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_iou = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, epoch)
        val_loss, val_iou = validate(model, val_loader, epoch)

        print(f"[Epoch {epoch}] "
              f"train_loss={train_loss:.6f}  train_IoU={train_iou:.6f}  "
              f"val_loss={val_loss:.6f}  val_IoU={val_iou:.6f}")

        # # save "last" checkpoint for this round
        # torch.save(
        #     {"model_state_dict": model.state_dict()},
        #     LAST_CKPT,
        # )

        # save best by val IoU
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {"model_state_dict": model.state_dict()},
                BEST_CKPT,
            )
            print(f"New best IoU: {best_iou:.6f}  (model saved to {BEST_CKPT})")

    print("Training done.")
    # print(f"Last checkpoint: {LAST_CKPT}")
    print(f"Best checkpoint: {BEST_CKPT} (IoU={best_iou:.6f})")


if __name__ == "__main__":
    main()
