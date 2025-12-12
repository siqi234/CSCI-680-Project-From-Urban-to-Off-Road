import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from transformers import SegformerForSemanticSegmentation
from dataset_ORFD import ORFDFinetuneDataset   # your dataset class

ROUND_NUM = 5  # 0(baseline), 1, 2, ...

# =================== CONFIG ===================
TEST_CSV = "OFRD/test_ORFD.csv"  # CSV with image_path, gt_path

# MODEL_PATH = "models/segformer/segformer_orfd_full.pth"
# MODEL_PATH = f"models/segformer/segformer_drivable_best.pth"
MODEL_PATH = f"models/segformer/segformer_orfd_round{ROUND_NUM}.pth"

# Base HF model you originally used for training
BASE_MODEL_NAME = "nvidia/segformer-b1-finetuned-ade-512-512"

IMG_SIZE = (352, 640)   # (H, W)
BATCH_SIZE = 2
NUM_WORKERS = 4
NUM_DISPLAY = 5          # number of samples to show

NUM_CLASSES = 2          # [background, drivable]
DRIVABLE_CLASS = 1       # index of drivable in logits
THRESH = 0.5             # prob threshold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==============================================


# --- Metric functions ---
def calc_iou(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)


def calc_dice(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def calc_precision(pred, target, eps=1e-6):
    """Precision = TP / (TP + FP)"""
    pred = pred.view(-1)
    target = target.view(-1)
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp + eps) / (tp + fp + eps)


def calc_recall(pred, target, eps=1e-6):
    """Recall = TP / (TP + FN)"""
    pred = pred.view(-1)
    target = target.view(-1)
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + eps) / (tp + fn + eps)


def calc_f1(pred, target, eps=1e-6):
    """F1 = 2 * (precision * recall) / (precision + recall)"""
    precision = calc_precision(pred, target, eps)
    recall = calc_recall(pred, target, eps)
    return (2 * precision * recall + eps) / (precision + recall + eps)


# --- Visualization ---
def show_predictions(images, masks_gt, masks_pred, num_display=5):
    num_display = min(num_display, images.size(0))
    plt.figure(figsize=(15, 5 * num_display))
    for i in range(num_display):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        gt = masks_gt[i, 0].cpu().numpy()
        pred = masks_pred[i, 0].cpu().numpy()

        plt.subplot(num_display, 3, i * 3 + 1)
        plt.imshow(img)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(num_display, 3, i * 3 + 2)
        plt.imshow(gt, cmap="gray", vmin=0, vmax=1)
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(num_display, 3, i * 3 + 3)
        plt.imshow(pred, cmap="gray", vmin=0, vmax=1)
        plt.title("Prediction")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def build_segformer(num_classes: int, ckpt_path: str, base_model_name: str):
    """
    If ckpt_path is a directory -> assume it's a HF 'save_pretrained' folder.
    If ckpt_path is a file (.pth) -> load as state_dict on top of base_model_name.
    """
    if os.path.isdir(ckpt_path):
        print(f"[INFO] Loading SegFormer from directory: {ckpt_path}")
        model = SegformerForSemanticSegmentation.from_pretrained(
            ckpt_path,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        return model

    # Otherwise, treat ckpt_path as a .pth file with weights
    print(f"[INFO] Loading base SegFormer from HF: {base_model_name}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        base_model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    print(f"[INFO] Loading state_dict from checkpoint file: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")

    # If saved as {'model_state_dict': ...}
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint ({len(missing)}): {missing[:10]}...")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint ({len(unexpected)}): {unexpected[:10]}...")

    return model


def main():
    print("Loading ORFD test dataset...")

    # Load full ORFD test dataset from CSV
    full_dataset = ORFDFinetuneDataset(
        csv_path=TEST_CSV,
        size=IMG_SIZE,
    )

    test_loader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(f"Evaluating on ALL {len(full_dataset)} ORFD test images...")

    # ---- Model ----
    model = build_segformer(NUM_CLASSES, MODEL_PATH, BASE_MODEL_NAME).to(DEVICE)
    model.eval()

    # ---- Evaluation loop ----
    iou_total, dice_total = 0.0, 0.0
    precision_total, recall_total, f1_total = 0.0, 0.0, 0.0
    count = 0
    vis_imgs, vis_gt, vis_pred = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating (SegFormer)", ncols=100):
            # ORFDFinetuneDataset returns (img, mask) or (img, mask, extra)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                imgs, masks = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                imgs, masks, _ = batch
            else:
                raise ValueError("Unexpected batch format from ORFDFinetuneDataset.")

            imgs = imgs.to(DEVICE)     # treat as pixel_values
            masks = masks.to(DEVICE)   # (B,1,H,W) or (B,H,W)

            # Ensure masks shape is (B,1,H,W)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)

            # Forward pass: SegFormer
            outputs = model(pixel_values=imgs)
            logits = outputs.logits  # (B, C, h', w')

            # Upsample logits to match input size
            logits_up = F.interpolate(
                logits,
                size=imgs.shape[-2:],   # (H, W)
                mode="bilinear",
                align_corners=False,
            )

            probs_all = torch.softmax(logits_up, dim=1)  # (B, C, H, W)
            drivable_probs = probs_all[:, DRIVABLE_CLASS:DRIVABLE_CLASS + 1, :, :]  # (B,1,H,W)
            preds = (drivable_probs > THRESH).float()

            batch_size = imgs.size(0)

            for b in range(batch_size):
                iou = calc_iou(preds[b], masks[b])
                dice = calc_dice(preds[b], masks[b])
                precision = calc_precision(preds[b], masks[b])
                recall = calc_recall(preds[b], masks[b])
                f1 = calc_f1(preds[b], masks[b])

                iou_total += iou.item()
                dice_total += dice.item()
                precision_total += precision.item()
                recall_total += recall.item()
                f1_total += f1.item()
                count += 1

                # store a few examples for visualization
                if len(vis_imgs) < NUM_DISPLAY:
                    vis_imgs.append(imgs[b].cpu())
                    vis_gt.append(masks[b].cpu())
                    vis_pred.append(preds[b].cpu())

    mean_iou = iou_total / count
    mean_dice = dice_total / count
    mean_precision = precision_total / count
    mean_recall = recall_total / count
    mean_f1 = f1_total / count

    print("\nSegFormer Evaluation on FULL ORFD test set:")
    print(f"Mean IoU:       {mean_iou:.6f}")
    print(f"Mean Dice:      {mean_dice:.6f}")
    print(f"Mean Precision: {mean_precision:.6f}")
    print(f"Mean Recall:    {mean_recall:.6f}")
    print(f"Mean F1:        {mean_f1:.6f}")

    # ---- Visualization ----
    if vis_imgs:
        print(f"\nShowing {len(vis_imgs)} predictions (SegFormer):")
        show_predictions(
            torch.stack(vis_imgs),
            torch.stack(vis_gt),
            torch.stack(vis_pred),
            num_display=NUM_DISPLAY
        )


if __name__ == "__main__":
    main()