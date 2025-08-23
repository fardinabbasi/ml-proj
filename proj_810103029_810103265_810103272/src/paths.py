from pathlib import Path
import os

# Set once in notebook if needed:
# os.environ["DATA_DIR"] = "/content/dataset"
DATA_DIR = Path(os.environ.get("DATA_DIR", "/content/dataset")).resolve()

train_images = DATA_DIR / "train" / "images"
train_labels = DATA_DIR / "train" / "labels"
val_images   = DATA_DIR / "val"   / "images"
val_labels   = DATA_DIR / "val"   / "labels"
test_images  = DATA_DIR / "test"  / "images"
test_labels  = DATA_DIR / "test"  / "labels"

aug_images   = DATA_DIR / "train_aug" / "images"
aug_labels   = DATA_DIR / "train_aug" / "labels"

# keep your variable spelling
pseudu_label = DATA_DIR / "pseudo_labels"

def ensure_dirs():
    for p in [train_images, train_labels, val_images, val_labels, test_images, test_labels, aug_images, aug_labels, pseudu_label]:
        p.mkdir(parents=True, exist_ok=True)