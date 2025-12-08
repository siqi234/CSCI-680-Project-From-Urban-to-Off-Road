import os
import csv
import numpy as np
from pathlib import Path
from PIL import Image

ROUND_NUM = 1  
# ============ CONFIG ============
IMG_DIR = Path("OFRD/train/image_data")

PROB_DIRS = [
    Path(f"prediction/round{ROUND_NUM}/unet"),
    Path(f"prediction/round{ROUND_NUM}/deeplabv3++"),
    Path(f"prediction/round{ROUND_NUM}/segformer"),
]

OUT_CSV = f"prediction/round{ROUND_NUM}/attention_scores_round{ROUND_NUM}.csv"

EPS = 1e-6
THRESH_MASK = 0.5

# weights for overall attention
W_INC = 1.0
W_UNC = 1.0
W_BC  = 1.0
# ================================


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3]


def resize_prob_to(prob, target_shape):
    th, tw = target_shape
    im = Image.fromarray((prob * 255).astype(np.uint8))
    im = im.resize((tw, th), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def load_probs(prob_paths, target_shape):
    probs = []
    for p in prob_paths:
        arr = np.load(p)  # [H,W]
        if arr.ndim != 2:
            raise ValueError(f"{p} is not 2D, got {arr.shape}")
        if arr.shape != target_shape:
            arr = resize_prob_to(arr, target_shape)
        probs.append(arr.astype(np.float32))
    return np.stack(probs, axis=0)  # [N,H,W]


def compute_attention_score(img_path, prob_paths):
    """
    Compute attention score for a single image,
    using the same formula as in your single-image script.
    """
    # ----- load image -----
    img = load_image(img_path)
    H, W, _ = img.shape
    target_shape = (H, W)

    # ----- load & align probs -----
    probs = load_probs(prob_paths, target_shape)  # [N,H,W]
    N = probs.shape[0]

    # mean/min/max prob over models
    mean_p = probs.mean(axis=0)
    max_p = probs.max(axis=0)
    min_p = probs.min(axis=0)

    # ========== 1) INCONSISTENCY (model disagreement) ==========
    inconsistency = np.sqrt(((probs - mean_p) ** 2).mean(axis=0))

    # ========== 2) UNCERTAINTY (entropy of ensemble mean) ==========
    pm = np.clip(mean_p, EPS, 1.0 - EPS)
    uncertainty = -(pm * np.log(pm) + (1.0 - pm) * np.log(1.0 - pm))

    # ========== 3) BOUNDARY CONFLICT ==========
    diff = max_p - min_p
    gy, gx = np.gradient(mean_p)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_mag /= (grad_mag.max() + 1e-6)
    bc = diff * (grad_mag ** 0.5)

    # ---------- REGION OF INTEREST (for scoring) ----------
    union_pred = (probs >= THRESH_MASK).any(axis=0)
    ambiguous = (mean_p > 0.2) & (mean_p < 0.8)
    roi = (union_pred | ambiguous).astype(np.float32)

    # ---------- SCORE FOR SELECTION ----------
    def safe_scale(x):
        m = x.max()
        return x / (m + 1e-6) if m > 1e-8 else np.zeros_like(x)

    inc_s = safe_scale(inconsistency)
    unc_s = safe_scale(uncertainty)
    bc_s  = safe_scale(bc)

    att_raw = W_INC * inc_s + W_UNC * unc_s + W_BC * bc_s
    att_raw *= roi  # ignore confident background

    if roi.sum() > 0:
        attention_score = float(att_raw.sum() / roi.sum())
    else:
        attention_score = 0.0

    return attention_score


def find_image_path(stem, img_dir):
    """
    Given a stem like '1619778700874', try a list of extensions
    and return the first existing path.
    """
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    for ext in exts:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    # use the first model's folder as the reference list of stems
    ref_dir = PROB_DIRS[0]
    npy_files = [f for f in os.listdir(ref_dir) if f.endswith(".npy")]

    results = []

    print(f"Found {len(npy_files)} images to score.\n")

    from tqdm import tqdm

    # progress bar added here
    for fname in tqdm(npy_files, desc="Computing attention scores", ncols=100):
        stem = Path(fname).stem  # e.g. "1619778700874"

        img_path = find_image_path(stem, IMG_DIR)
        if img_path is None:
            continue

        prob_paths = [d / fname for d in PROB_DIRS]
        if not all(p.exists() for p in prob_paths):
            continue

        score = compute_attention_score(img_path, prob_paths)
        results.append((str(img_path), score))

    if not results:
        print("No scores computed. Check directories and filenames.")
        return

    results.sort(key=lambda x: x[1], reverse=True)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "attention_score"])
        writer.writerows(results)

    print(f"\nSaved {len(results)} scores to {OUT_CSV}")
    print("\nTop 10 most informative images:")
    for img_path, score in results[:10]:
        print(f"{score:.6f}  |  {img_path}")


if __name__ == "__main__":
    main()
