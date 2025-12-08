# dataset_ORFD.py

import csv
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class ORFDFinetuneDataset(Dataset):
    """
    Dataset for finetuning on OFRD using CSVs that already contain:
        image_path, gt_path

    CSV format:
    -----------
    image_path,gt_path
    OFRD\\train\\image_data\\1623....png,OFRD\\train\\gt_image\\1623...._fillcolor.png
    ...

    Returns:
        img_tensor  : (3, H, W), float32, normalized (ImageNet)
        mask_tensor : (1, H, W), float32 in {0,1}
    """

    def __init__(self, csv_path, size):
        """
        Parameters
        ----------
        csv_path : str or Path
            Path to CSV with columns [image_path, gt_path].
        size : (int, int)
            (H, W) target size for images and masks.
        """
        self.csv_path = Path(csv_path)
        self.h, self.w = size

        assert self.csv_path.exists(), f"CSV not found: {self.csv_path}"

        self.samples = self._read_csv(self.csv_path)

        if not self.samples:
            raise ValueError(f"No valid image paths found in {self.csv_path}")

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _read_csv(self, csv_path):
        samples = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if "image_path" not in reader.fieldnames or "gt_path" not in reader.fieldnames:
                raise ValueError(f"CSV {csv_path} must have columns: image_path, gt_path")

            for row in reader:
                img_path = Path(row["image_path"])
                gt_path  = Path(row["gt_path"])

                # resolve relative paths (relative to project root)
                if not img_path.is_absolute():
                    img_path = img_path.resolve()
                if not gt_path.is_absolute():
                    gt_path = gt_path.resolve()

                if not img_path.exists():
                    print(f"[WARN] Image not found, skipping: {img_path}")
                    continue
                if not gt_path.exists():
                    print(f"[WARN] GT not found, skipping : {gt_path}")
                    continue

                samples.append((img_path, gt_path))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]

        # ----- image -----
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.w, self.h))  # (W, H)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std
        img_np = np.transpose(img_np, (2, 0, 1))  # (C,H,W)
        img_tensor = torch.from_numpy(img_np)      # float32

        # ----- mask -----
        gt = Image.open(gt_path).convert("L")
        gt = gt.resize((self.w, self.h), Image.NEAREST)
        gt_np = np.array(gt)

        # Only white (255) is drivable
        gt_bin = (gt_np == 255).astype(np.float32)
        mask_tensor = torch.from_numpy(gt_bin).unsqueeze(0)  # (1,H,W)

        return img_tensor, mask_tensor
