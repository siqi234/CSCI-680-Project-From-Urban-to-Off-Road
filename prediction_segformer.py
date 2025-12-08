import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from transformers import SegformerForSemanticSegmentation

from dataset_ORFD import ORFDFinetuneDataset


# =================== CONFIG ===================
ROUND_NUM = 1  # 0(baseline), 1, 2, ...

CSV_PATH = f"Unlabeled/unlabeled_pool_round{ROUND_NUM}.csv"   # CSV with image_path (and maybe gt_path)

# Base HF model you started from when training
BASE_MODEL_NAME = "nvidia/segformer-b1-finetuned-ade-512-512"

# Your fine-tuned checkpoint (PyTorch state dict)
# MODEL_PATH = f"models/segformer/segformer_drivable_best.pth"
MODEL_PATH = f"models/segformer/segformer_orfd_round{ROUND_NUM}.pth"


IMG_SIZE = (352, 640)   # (H, W) – must match your training / dataset resize
BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_DISPLAY = 5
THRESH = 0.5            # threshold on drivable prob

OUT_NPY_DIR = Path(f"prediction/round{ROUND_NUM}/segformer")   # probability maps (.npy)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 2          # SegFormer output channels
DRIVABLE_CLASS = 1       # index of drivable class in logits
# ==============================================


def build_segformer(num_classes: int):
    """
    Build SegFormer and load your fine-tuned .pth weights.

    1) Load base SegFormer from HuggingFace (same as training).
    2) Load state_dict from MODEL_PATH (supports both raw state_dict
       or {'model_state_dict': ...} formats).
    """
    print(f"[INFO] Loading base SegFormer from HF: {BASE_MODEL_NAME}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    ckpt_path = Path(MODEL_PATH)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"SegFormer checkpoint not found: {ckpt_path}")

    print(f"[INFO] Loading fine-tuned weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[WARN] Missing keys when loading checkpoint ({len(missing_keys)}):")
        print(missing_keys[:10], "..." if len(missing_keys) > 10 else "")
    if unexpected_keys:
        print(f"[WARN] Unexpected keys when loading checkpoint ({len(unexpected_keys)}):")
        print(unexpected_keys[:10], "..." if len(unexpected_keys) > 10 else "")

    return model


def show_samples(images, preds, names, num_display):
    num_display = min(num_display, len(images))
    plt.figure(figsize=(10, 4 * num_display))
    for i in range(num_display):
        img = images[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        pd_mask = preds[i][0].numpy()

        plt.subplot(num_display, 2, 2 * i + 1)
        plt.imshow(img)
        plt.title(names[i])
        plt.axis("off")

        plt.subplot(num_display, 2, 2 * i + 2)
        plt.imshow(pd_mask, cmap="gray", vmin=0, vmax=1)
        plt.title("Pred mask (binary)")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # ---- read image paths from CSV (for naming) ----
    df = pd.read_csv(CSV_PATH)
    if "image_path" not in df.columns:
        raise ValueError(f"'image_path' column not found in {CSV_PATH}")
    img_paths = df["image_path"].tolist()

    # ---- build dataset & loader using your existing dataset ----
    dataset = ORFDFinetuneDataset(
        csv_path=CSV_PATH,
        size=IMG_SIZE,
    )
    print(f"Found {len(dataset)} images from CSV.")

    if len(dataset) != len(img_paths):
        print(f"[WARN] dataset length ({len(dataset)}) != CSV rows ({len(img_paths)}). "
              f"Indexing assumes same order and length.")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,   # VERY IMPORTANT to keep alignment with CSV
        num_workers=NUM_WORKERS,
    )

    # ---- SegFormer model ----
    model = build_segformer(NUM_CLASSES).to(DEVICE)
    model.eval()
    print(f"Loaded SegFormer weights from: {MODEL_PATH}")

    OUT_NPY_DIR.mkdir(parents=True, exist_ok=True)

    vis_imgs, vis_preds, vis_names = [], [], []

    global_idx = 0  # to track which row in CSV we're on

    with torch.no_grad():
        for batch in tqdm(loader, ncols=100):
            # ORFDFinetuneDataset returns (img, mask) or (img, mask, extra)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                imgs, _ = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                imgs, _, _ = batch
            else:
                imgs = batch

            imgs = imgs.to(DEVICE)   # treated as pixel_values

            # Forward pass: HF SegFormer
            outputs = model(pixel_values=imgs)
            logits = outputs.logits                        # (B, C, h', w')

            # Upsample logits to match input spatial size (IMG_SIZE)
            logits_up = torch.nn.functional.interpolate(
                logits,
                size=imgs.shape[-2:],   # (H, W)
                mode="bilinear",
                align_corners=False,
            )

            probs_all = torch.softmax(logits_up, dim=1)    # (B, C, H, W)
            drivable_probs = probs_all[:, DRIVABLE_CLASS:DRIVABLE_CLASS+1, :, :]  # (B,1,H,W)
            preds = (drivable_probs > THRESH).float()

            batch_size = imgs.size(0)

            for b in range(batch_size):
                if global_idx >= len(img_paths):
                    break  # safety

                prob_np = drivable_probs[b, 0].cpu().numpy().astype(np.float32)  # [H,W]
                pred_np = (prob_np > THRESH).astype(np.uint8)                    # [H,W], 0/1

                # use CSV image_path for naming
                img_path = img_paths[global_idx]
                stem = Path(str(img_path)).stem

                npy_path = OUT_NPY_DIR / f"{stem}.npy"
                np.save(npy_path, prob_np)

                # collect a few for visualization
                if len(vis_imgs) < NUM_DISPLAY:
                    vis_imgs.append(imgs[b].cpu())
                    vis_preds.append(torch.from_numpy(pred_np).unsqueeze(0))
                    vis_names.append(stem)

                global_idx += 1

    print(f"Saved NPY prob maps to: {OUT_NPY_DIR}")

    if vis_imgs:
        print(f"\nShowing {len(vis_imgs)} example predictions (SegFormer)...")
        show_samples(vis_imgs, vis_preds, vis_names, NUM_DISPLAY)


if __name__ == "__main__":
    main()
