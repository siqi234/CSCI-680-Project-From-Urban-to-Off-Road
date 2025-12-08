import json
from pathlib import Path
import csv

import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def load_drivable_polygons_from_json(json_path):
    """
    Read one BDD100K-style json and return:
        img_stem, [ [ (x1,y1), (x2,y2), ... ], ... ]
    Only keep category == 'area/drivable'.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_stem = data["name"]
    polygons = []

    for frame in data.get("frames", []):
        for obj in frame.get("objects", []):
            if obj.get("category") == "area/drivable" and "poly2d" in obj:
                pts = []
                for pt in obj["poly2d"]:
                    # pt like [x, y, 'L'] or [x, y, ...]
                    x, y = float(pt[0]), float(pt[1])
                    pts.append((x, y))
                if len(pts) >= 3:
                    polygons.append(pts)

    return img_stem, polygons


def polygons_to_mask(polygons, img_size):
    """
    polygons: list of [(x,y), ...]
    img_size: (W, H)
    returns: (H, W) uint8 mask with {0,1}
    """
    W, H = img_size
    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)

    for poly in polygons:
        if len(poly) >= 3:
            draw.polygon(poly, outline=1, fill=1)

    return np.array(mask_img, dtype=np.uint8)


class BDDDrivableDataset(Dataset):
    def __init__(self,
                 csv_path,
                 img_root,
                 label_root,
                 size=(360, 640)):
        """
        csv_path: CSV with at least "image_name" column (stem, no extension)
                  If you also have has_drivable, it's fine; we just ignore 0-rows when building.
        img_root: directory with images for this split
        label_root: directory with JSON labels for this split
        size: (H, W) for training (keep 16:9-ish, e.g. (360, 640))
        """
        self.img_root = Path(img_root)
        self.label_root = Path(label_root)
        self.size = size

        self.items = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # If your CSV already filtered has_drivable==1, no check needed.
                # If not, uncomment:
                # if "has_drivable" in r and r["has_drivable"] != "1":
                #     continue
                self.items.append(r)

        if not self.items:
            raise ValueError(f"No items loaded from {csv_path}")

        self.img_transform = T.Compose([
            T.Resize(self.size),  # (H, W)
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        r = self.items[idx]

        # If CSV stores only stem:
        img_stem = r["image_name"]

        # If your CSV instead stores full path like BDD100k/train/xxx.jpg,
        # use: img_path = Path(r["image_path"]) and derive stem via img_path.stem.

        # --- JSON ---
        json_path = self.label_root / f"{img_stem}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Label JSON not found: {json_path}")

        img_stem_from_json, polygons = load_drivable_polygons_from_json(json_path)

        # Sanity: names should match; if not, trust JSON's name
        if img_stem_from_json:
            img_stem = img_stem_from_json

        # --- Image ---
        img_path = self.img_root / f"{img_stem}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # --- Mask in original size ---
        mask_np = polygons_to_mask(polygons, (W, H))  # (H, W), 0/1

        # --- Transform image ---
        img = self.img_transform(img)

        # --- Resize mask with NEAREST & to tensor ---
        # self.size is (H, W)
        h, w = self.size

        mask_pil = Image.fromarray(mask_np)
        # PIL expects (width, height)
        mask_pil = mask_pil.resize((w, h), resample=Image.NEAREST)

        mask_resized = np.array(mask_pil, dtype=np.float32)  # shape (H, W)
        mask = torch.from_numpy(mask_resized).unsqueeze(0)   # (1, H, W)

        return img, mask
