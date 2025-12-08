import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import pandas as pd

from dataset_ORFD import ORFDFinetuneDataset  # <-- use your existing dataset

# =================== CONFIG ===================
ROUND_NUM = 1  # 0(baseline), 1, 2, ...

# MODEL_PATH = "unet/unet_drivable_best.pth"
MODEL_PATH = f"unet/unet_ofrd_round{ROUND_NUM}.pth"

CSV_PATH = f"Unlabeled/unlabeled_pool_round{ROUND_NUM}.csv"   # CSV with image_path (and maybe gt_path)

IMG_SIZE = (352, 640)   # (H, W) must match your training
BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_DISPLAY = 5
THRESH = 0.5

OUT_NPY_DIR = Path(f"prediction/round{ROUND_NUM}/unet")   # probability maps (.npy)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==============================================


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

    # ---- model (same arch as training) ----
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    OUT_NPY_DIR.mkdir(parents=True, exist_ok=True)

    vis_imgs, vis_preds, vis_names = [], [], []

    global_idx = 0  # to track which row in CSV we're on

    with torch.no_grad():
        for imgs_batch in tqdm(loader, ncols=100):
            # ORFDTestDataset likely returns (img, mask) -> imgs_batch is a tuple
            if isinstance(imgs_batch, (list, tuple)) and len(imgs_batch) == 2:
                imgs, _ = imgs_batch
            elif isinstance(imgs_batch, (list, tuple)) and len(imgs_batch) == 3:
                imgs, _, _ = imgs_batch
            else:
                # if it's just imgs, handle that too
                imgs = imgs_batch

            imgs = imgs.to(DEVICE)

            logits = model(imgs)
            probs = torch.sigmoid(logits)         # (B,1,H,W)
            preds = (probs > THRESH).float()      # binary mask

            batch_size = imgs.size(0)

            for b in range(batch_size):
                if global_idx >= len(img_paths):
                    break  # safety

                prob_np = probs[b, 0].cpu().numpy().astype(np.float32)  # [H,W], prob
                pred_np = (prob_np > THRESH).astype(np.uint8)           # [H,W], 0/1

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
        print(f"\nShowing {len(vis_imgs)} example predictions...")
        show_samples(vis_imgs, vis_preds, vis_names, NUM_DISPLAY)


if __name__ == "__main__":
    main()