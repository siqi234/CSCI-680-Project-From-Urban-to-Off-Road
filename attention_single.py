import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from PIL import Image
import cv2
import os

IMG_NAME = '1623170282031'
ROUND_NUM = 4  # 0(baseline), 1, 2, ...
# ============ CONFIG ============
IMG_PATH = f"OFRD/train/image_data/{IMG_NAME}.png"        # original image
GT_PATH  = f"OFRD/train/gt_image/{IMG_NAME}_fillcolor.png"  # <<< NEW: GT mask path

PROB_PATHS = [
    f"prediction/round{ROUND_NUM}/unet/{IMG_NAME}.npy",        # U-Net prob
    f"prediction/round{ROUND_NUM}/deeplabv3++/{IMG_NAME}.npy", # DeepLabV3+ prob
    f"prediction/round{ROUND_NUM}/segformer/{IMG_NAME}.npy",   # SegFormer prob
]

ALPHA = 0.6
EPS = 1e-6

# thresholds for visualization overlays (NOT for logic)
TH_INC = 0.2
TH_UNC = 0.2
TH_BC  = 0.2
TH_ATT = 0.2

# weights for overall attention
W_INC = 1.0
W_UNC = 1.0
W_BC  = 1.0

# mask + contour settings
THRESH_MASK = 0.5
LINE_THICKNESS = 4
COLOR_M1 = (0.0, 0.47, 0.95)   # Blue - U-Net
COLOR_M2 = (1.0, 0.55, 0.0)    # Orange - DeepLabV3++
COLOR_M3 = (0.75, 0.1, 0.75)   # Magenta - SegFormer

GT_COLOR = (0.0, 0.0, 0.0)     # <<< NEW: Bright green for GT contour (RGB)
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
        arr = np.load(p)  # [H,W] with prob of drivable area
        if arr.ndim != 2:
            raise ValueError(f"{p} is not 2D, got {arr.shape}")
        if arr.shape != target_shape:
            arr = resize_prob_to(arr, target_shape)
        probs.append(arr.astype(np.float32))
    return np.stack(probs, axis=0)  # [N,H,W]


def normalize_map(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def get_cmap(name="turbo"):
    if name == "custom":
        colors = [
            (0.0, 0.0, 0.5),
            (0.0, 0.5, 1.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        ]
        return ListedColormap(colors)
    return plt.get_cmap(name)


def overlay_heat(base_img, score, thresh, alpha=0.6, cmap_name="turbo"):
    score = np.nan_to_num(score, nan=0.0)

    # robust norm: visualization only
    vmin = np.percentile(score, 5)
    vmax = np.percentile(score, 99)
    if vmax <= vmin:
        return base_img.copy()

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = get_cmap(cmap_name)
    heat = cmap(norm(score))[..., :3]

    mask = score > thresh

    out = base_img.copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * heat[mask]
    return np.clip(out, 0, 1)


def draw_contours_on_top(img, probs, colors, thresh=0.5, thickness=3):
    img_uint8 = (img * 255).astype(np.uint8)
    for prob, color in zip(probs, colors):
        mask = (prob >= thresh).astype(np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # RGB -> BGR for OpenCV
        bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
        cv2.drawContours(img_uint8, contours, -1, bgr, thickness)
    return img_uint8.astype(np.float32) / 255.0


# <<< NEW: load GT mask and draw GT contour on original image >>>
def load_gt_mask(gt_path, target_shape):
    """
    Load GT mask (fillcolor style), resize to target_shape (H, W),
    and return binary mask {0,1}.
    """
    if not os.path.isfile(gt_path):
        print(f"[WARN] GT mask not found: {gt_path}")
        return None

    mask = Image.open(gt_path).convert("L")
    h, w = target_shape
    mask = mask.resize((w, h), resample=Image.NEAREST)
    mask_np = np.array(mask)
    # treat non-zero as drivable
    mask_bin = (mask_np > 0).astype(np.uint8)
    return mask_bin


def draw_gt_contour_on_img(img, gt_mask, color=(0.0, 1.0, 0.0), thickness=3):
    """
    Draw GT contour on top of the original RGB image.
    img: [H,W,3] float in [0,1]
    gt_mask: [H,W] uint8 in {0,1}
    color: RGB tuple in [0,1]
    """
    img_uint8 = (img * 255).astype(np.uint8)
    mask_uint8 = (gt_mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # RGB -> BGR
    bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
    cv2.drawContours(img_uint8, contours, -1, bgr, thickness)

    return img_uint8.astype(np.float32) / 255.0
# <<< NEW END >>>


def main():
    # ----- load image -----
    img = load_image(IMG_PATH)
    H, W, _ = img.shape
    target_shape = (H, W)

    # ----- load GT & draw GT contour on original image -----
    gt_mask = load_gt_mask(GT_PATH, target_shape)    # <<< NEW
    if gt_mask is not None:
        img_with_gt = draw_gt_contour_on_img(
            img, gt_mask, color=GT_COLOR, thickness=LINE_THICKNESS
        )
    else:
        img_with_gt = img.copy()
    # ^ this will be used as the first panel

    # ----- load & align probs -----
    probs = load_probs(PROB_PATHS, target_shape)  # [N,H,W]
    N = probs.shape[0]
    print(f"Loaded {N} prob maps at {target_shape}")

    # mean/min/max prob over models
    mean_p = probs.mean(axis=0)
    max_p = probs.max(axis=0)
    min_p = probs.min(axis=0)

    # ========== 1) INCONSISTENCY ==========
    inconsistency = np.sqrt(((probs - mean_p) ** 2).mean(axis=0))

    # ========== 2) UNCERTAINTY ==========
    pm = np.clip(mean_p, EPS, 1.0 - EPS)
    uncertainty = -(pm * np.log(pm) + (1.0 - pm) * np.log(1.0 - pm))

    # ========== 3) BOUNDARY CONFLICT ==========
    diff = max_p - min_p
    gy, gx = np.gradient(mean_p)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_mag /= (grad_mag.max() + 1e-6)
    bc = diff * (grad_mag ** 0.5)

    # ---------- REGION OF INTEREST ----------
    union_pred = (probs >= THRESH_MASK).any(axis=0)
    ambiguous = (mean_p > 0.2) & (mean_p < 0.8)
    roi = (union_pred | ambiguous).astype(np.float32)

    # ---------- SCORE ----------
    def safe_scale(x):
        m = x.max()
        return x / (m + 1e-6) if m > 1e-8 else np.zeros_like(x)

    inc_s = safe_scale(inconsistency)
    unc_s = safe_scale(uncertainty)
    bc_s  = safe_scale(bc)

    att_raw = W_INC * inc_s + W_UNC * unc_s + W_BC * bc_s
    att_raw *= roi

    if roi.sum() > 0:
        attention_score = float(att_raw.sum() / roi.sum())
    else:
        attention_score = 0.0

    print(f"Attention score for this image (0-3 range-ish): {attention_score:.6f}")

    # ---------- MAPS FOR VISUALIZATION ----------
    inc_n = normalize_map(inconsistency)
    unc_n = normalize_map(uncertainty)
    bc_n  = normalize_map(bc)
    att_n = normalize_map(att_raw)

    inc_heat = overlay_heat(img, inc_n, TH_INC, alpha=ALPHA, cmap_name="plasma")
    unc_heat = overlay_heat(img, unc_n, TH_UNC, alpha=ALPHA, cmap_name="plasma")
    bc_heat  = overlay_heat(img, bc_n,  TH_BC,  alpha=ALPHA, cmap_name="plasma")
    att_heat = overlay_heat(img, att_n, TH_ATT, alpha=ALPHA, cmap_name="plasma")

    # add model contours
    contour_colors = []
    if N >= 1:
        contour_colors.append(COLOR_M1)
    if N >= 2:
        contour_colors.append(COLOR_M2)
    if N >= 3:
        contour_colors.append(COLOR_M3)

    inc_vis = draw_contours_on_top(inc_heat, probs, contour_colors,
                                   thresh=THRESH_MASK, thickness=LINE_THICKNESS)
    unc_vis = draw_contours_on_top(unc_heat, probs, contour_colors,
                                   thresh=THRESH_MASK, thickness=LINE_THICKNESS)
    bc_vis  = draw_contours_on_top(bc_heat,  probs, contour_colors,
                                   thresh=THRESH_MASK, thickness=LINE_THICKNESS)
    att_vis = draw_contours_on_top(att_heat, probs, contour_colors,
                                   thresh=THRESH_MASK, thickness=LINE_THICKNESS)

    # ---------- PLOT PANEL WITH COLORBAR ----------
    fig, axes = plt.subplots(1, 5, figsize=(22, 4),
                             gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.05]})

    # use img_with_gt for the first panel
    panels = [img_with_gt, inc_vis, unc_vis, bc_vis, att_vis]
    titles = [
        "Input Image + GT Contour",
        "Inconsistency (Model Disagreement)",
        "Uncertainty (Entropy of Mean)",
        "Boundary Conflict",
        "Overall Attention (Ours)"
    ]

    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=11, weight="semibold")
        ax.axis("off")

    # ----- colorbar -----
    cmap = plt.get_cmap("plasma")
    norm = Normalize(vmin=0.0, vmax=1.0)
    cb_ax = fig.add_axes([0.91, 0.20, 0.015, 0.60])
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cb_ax, orientation='vertical')
    cbar.set_label('Attention Level', fontsize=10, weight='semibold', labelpad=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.invert_yaxis()

    # ----- legend for contours -----
    legend_elements = [
        Line2D([0], [0], color=COLOR_M1, lw=2, label=f'U-Net (round {ROUND_NUM})'),
        Line2D([0], [0], color=COLOR_M2, lw=2, label=f'DeepLabV3++ (round {ROUND_NUM})'),
        Line2D([0], [0], color=COLOR_M3, lw=2, label=f'SegFormer (round {ROUND_NUM})'),
        Line2D([0], [0], color=GT_COLOR, lw=2, label='Ground Truth'),   # <<< NEW
    ]
    fig.legend(handles=legend_elements,
               loc='lower center',
               ncol=4,
               bbox_to_anchor=(0.5, -0.02),
               fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 0.9, 1])
    plt.show()


if __name__ == "__main__":
    main()
