import pandas as pd
from pathlib import Path

# ======= CONFIG =======
ROUND_NUM = 4  # current round index, whose attention_scores we just computed

# attention scores from current unlabeled pool (already sorted by attention_score desc)
CSV_PATH_ATT = f"prediction/round{ROUND_NUM}/attention_scores_round{ROUND_NUM}.csv"

TOP_RATIO = 0.15  # top 15% go into training

# previous training pool (used to finetune round ROUND_NUM model)
PREV_TRAIN_CSV = f"Unlabeled/finetuning_pool_round{ROUND_NUM}.csv"

# new outputs (for next round)
OUT_TRAIN_NEXT = f"Unlabeled/finetuning_pool_round{ROUND_NUM+1}.csv"
OUT_UNLABELED_NEXT = f"Unlabeled/unlabeled_pool_round{ROUND_NUM+1}.csv"
# ======================


def compute_gt_path(img_path: str) -> str:
    """
    Convert image path → GT path using OFRD naming:
    .../image_data/xxxx.ext → .../gt_image/xxxx_fillcolor.ext
    """
    p = Path(img_path)
    stem = p.stem          # '1619778700874'
    ext = p.suffix         # '.png', '.jpg', ...
    parent = Path(str(p.parent).replace("image_data", "gt_image"))
    gt_path = parent / f"{stem}_fillcolor{ext}"
    return str(gt_path)


def main():
    # -------- 1) Read attention scores (current unlabeled pool) --------
    df_att = pd.read_csv(CSV_PATH_ATT)

    total = len(df_att)
    n_top = max(1, int(total * TOP_RATIO))

    print(f"Total unlabeled rows this round = {total}")
    print(f"Top {TOP_RATIO*100:.0f}% = {n_top} rows")

    # We assume df_att is already sorted by attention_score descending
    df_top = df_att.iloc[:n_top].copy()       # selected for labeling
    df_bottom = df_att.iloc[n_top:].copy()    # remain unlabeled

    # -------- 2) Ensure gt_path exists for both groups --------
    if "gt_path" not in df_top.columns:
        df_top["gt_path"] = df_top["image_path"].apply(compute_gt_path)
    if "gt_path" not in df_bottom.columns:
        df_bottom["gt_path"] = df_bottom["image_path"].apply(compute_gt_path)

    # Drop attention_score from both (not needed for training/unlabeled CSVs)
    for part in (df_top, df_bottom):
        if "attention_score" in part.columns:
            part.drop(columns=["attention_score"], inplace=True)

    # -------- 3) Read previous training pool and append df_top --------
    prev_train_df = pd.read_csv(PREV_TRAIN_CSV)

    # Make sure prev_train_df also has gt_path (if not already)
    if "gt_path" not in prev_train_df.columns:
        if "image_path" not in prev_train_df.columns:
            raise ValueError(f"'image_path' column not found in {PREV_TRAIN_CSV}")
        prev_train_df["gt_path"] = prev_train_df["image_path"].apply(compute_gt_path)

    # Concatenate: previous training pool + newly selected top 15%
    train_next = pd.concat([prev_train_df, df_top], ignore_index=True)

    # Optional: drop duplicates by image_path just in case
    train_next = train_next.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

    # -------- 4) Save new training pool and new unlabeled pool --------
    train_next.to_csv(OUT_TRAIN_NEXT, index=False)
    df_bottom.to_csv(OUT_UNLABELED_NEXT, index=False)

    print("\nSaved:")
    print(f"  New training pool:  {OUT_TRAIN_NEXT}   "
          f"(prev training + top 15% of current unlabeled)")
    print(f"  New unlabeled pool: {OUT_UNLABELED_NEXT} (remaining 85%)")

    print("\nSample from new training pool:")
    print(train_next.head())


if __name__ == "__main__":
    main()
