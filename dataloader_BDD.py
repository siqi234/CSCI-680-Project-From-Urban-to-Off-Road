# dataloaders.py (or inside your train script)

from torch.utils.data import DataLoader
from dataset_BDD import BDDDrivableDataset

train_dataset = BDDDrivableDataset(
    csv_path="train_list.csv",
    img_root="BDD100k/train",
    label_root="BDD100k_labels/train",
    # size=(360, 640),
    size=(720, 1280),  # (H, W)
)

val_dataset = BDDDrivableDataset(
    csv_path="val_list.csv",
    img_root="BDD100k/val",
    label_root="BDD100k_labels/val",
    # size=(360, 640),
    size=(720, 1280),  # (H, W)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
