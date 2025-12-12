import json
import csv
from pathlib import Path
from tqdm import tqdm

# ----- PATHS -----

# # Training images and labels directories
# img_dir = Path("BDD100k/train")
# label_dir = Path("BDD100k_labels/train")
# out_csv = Path("train_drivable_check.csv")

# # Valication images and labels directories
# img_dir = Path("BDD100k/val")
# label_dir = Path("BDD100k_labels/val")
# out_csv = Path("val_drivable_check.csv")

# Testing images and labels directories
img_dir = Path("BDD100k/test")
label_dir = Path("BDD100k_labels/test")
out_csv = Path("test_drivable_check.csv")

# -----------------

rows = []

# List all JSON files
json_files = sorted(label_dir.glob("*.json"))

# Loop with progress bar
for json_path in tqdm(json_files, desc="Checking drivable areas", ncols=100):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_name = data.get("name", json_path.stem)
    has_drivable = 0

    # Each file has frames -> objects
    for frame in data.get("frames", []):
        for obj in frame.get("objects", []):
            if obj.get("category") == "area/drivable":
                has_drivable = 1
                break
        if has_drivable:
            break

    rows.append({
        "image_name": img_name,
        "image_path": str(img_dir / f"{img_name}.jpg"),
        "has_drivable": has_drivable,
    })

# Write CSV
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image_name", "image_path", "has_drivable"])
    writer.writeheader()
    writer.writerows(rows)

# Print summary
total = len(rows)
has_any = sum(r["has_drivable"] for r in rows)
print(f"\n CSV saved to: {out_csv}")
print(f"Total JSON files checked: {total}")
print(f"Images with drivable area: {has_any} ({has_any / total * 100:.2f}%)")
