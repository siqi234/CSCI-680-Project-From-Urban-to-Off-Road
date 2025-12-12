import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from dataset_ORFD import ORFDFinetuneDataset   # your dataset class

ROUND_NUM = 5  # 0(baseline), 1, 2, ...

# =================== CONFIG ===================
TEST_CSV = "OFRD/test_ORFD.csv"  # CSV with image_path, gt_path

MODEL_PATH = f"models/unet/unet_ofrd_round{ROUND_NUM}.pth"
# MODEL_PATH = f"models/unet/unet_drivable_best.pth"
# MODEL_PATH = f"models/unet/unet_ofrd_full.pth"


IMG_SIZE = (352, 640)   # (H, W)
BATCH_SIZE = 2
NUM_WORKERS = 4
NUM_DISPLAY = 5          # number of samples to show

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
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # ---- Evaluation loop ----
    iou_total, dice_total = 0.0, 0.0
    precision_total, recall_total, f1_total = 0.0, 0.0, 0.0
    count = 0
    vis_imgs, vis_gt, vis_pred = [], [], []

    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Evaluating", ncols=100):

            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

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

    print("\nEvaluation Results on FULL ORFD test set:")
    print(f"Mean IoU:       {mean_iou:.6f}")
    print(f"Mean Dice:      {mean_dice:.6f}")
    print(f"Mean Precision: {mean_precision:.6f}")
    print(f"Mean Recall:    {mean_recall:.6f}")
    print(f"Mean F1:        {mean_f1:.6f}")

    # ---- Visualization ----
    if vis_imgs:
        print(f"\nShowing {len(vis_imgs)} predictions:")
        show_predictions(
            torch.stack(vis_imgs),
            torch.stack(vis_gt),
            torch.stack(vis_pred),
            num_display=NUM_DISPLAY
        )


if __name__ == "__main__":
    main()