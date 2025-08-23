# viz_crops.py
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms
from data_utils import load_image, load_label, crop_characters, get_x
from transforms_setup import train_transform
import os

def _find_dataset_root() -> Path:
    # Try DATA_DIR
    dd = os.environ.get("DATA_DIR")
    if dd and (Path(dd) / "train" / "images").exists():
        return Path(dd)

    # Try PROJECT_ROOT/dataset
    pr = os.environ.get("PROJECT_ROOT")
    if pr and (Path(pr) / "dataset" / "train" / "images").exists():
        return Path(pr) / "dataset"

    # Walk up from this file
    here = Path(__file__).resolve()
    for up in [here.parent, *here.parents]:
        cand = up / "dataset"
        if (cand / "train" / "images").exists():
            return cand

    # Search Drive
    drive_base = Path("/content/drive/MyDrive")
    if drive_base.exists():
        for cand in drive_base.glob("**/dataset"):
            if (cand / "train" / "images").exists():
                return cand

    raise FileNotFoundError("Could not locate dataset root.")

def _denorm_gray(t):
    t = t * 0.5 + 0.5
    t = t.clamp(0, 1)
    return transforms.ToPILImage()(t)

def plot_all_char_views(image_id, image_dir, label_dir, n_views=10):
    # If the provided dirs don't exist, resolve to the correct dataset root
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    if not image_dir.exists() or not label_dir.exists():
        root = _find_dataset_root()
        image_dir = root / "train" / "images"
        label_dir = root / "train" / "labels"
        print(f"[viz] Auto-corrected paths → {image_dir}, {label_dir}")

    img_path  = image_dir / f"{image_id}.png"
    if not img_path.exists():
        alt = image_dir / f"{image_id}.jpg"
        if alt.exists():
            img_path = alt
    json_path = label_dir / f"{image_id}.json"

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Label not found: {json_path}")

    pil_img = load_image(img_path)
    bboxes, expr = load_label(json_path)
    if not bboxes:
        raise ValueError(f"No bboxes in {json_path.name}")

    bboxes = sorted(bboxes, key=get_x)
    crops  = crop_characters(pil_img, bboxes)

    if isinstance(expr, list):
        labels = [c if isinstance(c, str) and c else "?" for c in expr]
    else:
        expr = expr or ""
        labels = list(expr) + ["?"] * max(0, len(crops) - len(expr))

    rows, cols = len(crops), n_views
    plt.figure(figsize=(cols * 2.0, rows * 2.2))
    k = 1
    for r, crop in enumerate(crops):
        views = [_denorm_gray(train_transform(crop)) for _ in range(n_views)]
        for c in range(n_views):
            ax = plt.subplot(rows, cols, k); k += 1
            ax.imshow(views[c], cmap="gray"); ax.axis("off")
            if c == 0:
                ax.set_title(f"#{r} '{labels[r] if r < len(labels) else '?'}'", fontsize=10, pad=6)
    plt.suptitle(f"ID {image_id} — {n_views} views per character", y=0.995)
    plt.tight_layout()
    plt.show()
