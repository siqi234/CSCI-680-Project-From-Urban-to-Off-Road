# finetune_unet.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
from pathlib import Path

from dataset_ORFD import ORFDFinetuneDataset

# ROUND_NUM = 1  # 0(baseline), 1, 2, ...
# ================== CONFIG ==================
TRAIN_CSV = f"Unlabeled/unlabeled_pool_full.csv"
VAL_CSV   = f"OFRD/validation_ORFD.csv"   # or wherever you saved it

IMG_SIZE    = (352, 640)   # (H, W)
BATCH_SIZE  = 4
EPOCHS      = 15
LR          = 1e-5
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRETRAINED_WEIGHTS = "unet/unet_drivable_best.pth"
FINETUNE_DIR = Path("unet")
FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
# ============================================


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
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Finetune Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    return train_loader, val_loader


def build_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    if PRETRAINED_WEIGHTS and Path(PRETRAINED_WEIGHTS).exists():
        state_dict = torch.load(PRETRAINED_WEIGHTS, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from: {PRETRAINED_WEIGHTS}")
    else:
        print("[WARN] PRETRAINED_WEIGHTS not found; finetuning from scratch.")

    return model


def calc_iou(preds, targets, eps=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    if union == 0:
        return torch.tensor(1.0, device=preds.device)
    return (intersection + eps) / (union + eps)


def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0

    for imgs, masks in tqdm(loader, desc=f"Epoch {epoch} [finetune_train]", ncols=100):
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    n_samples = 0

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=f"Epoch {epoch} [finetune_val]", ncols=100):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(imgs)
            loss = criterion(logits, masks)
            running_loss += loss.item() * imgs.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            batch_iou = calc_iou(preds, masks)
            running_iou += batch_iou.item() * imgs.size(0)
            n_samples += imgs.size(0)

    avg_loss = running_loss / len(loader.dataset)
    avg_iou = running_iou / max(n_samples, 1)
    return avg_loss, avg_iou


def main():
    train_loader, val_loader = build_loaders()
    model = build_model()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_iou = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_iou = validate(model, val_loader, criterion, epoch)

        print(f"[Finetune Epoch {epoch}] "
              f"train_loss={train_loss:.6f}  "
              f"val_loss={val_loss:.6f}  "
              f"val_IoU={val_iou:.6f}")

        # # save latest
        # latest_path = FINETUNE_DIR / f"unet_ofrd_finetune_epoch{epoch}.pth"
        # torch.save(model.state_dict(), latest_path)

        # save best
        if val_iou > best_iou:
            best_iou = val_iou
            best_path = FINETUNE_DIR / f"unet_ofrd_full.pth"
            torch.save(model.state_dict(), best_path)
            print(f"New best IoU (finetune): {best_iou:.6f} (saved to {best_path})")

    print("Finetuning done.")


if __name__ == "__main__":
    main()
