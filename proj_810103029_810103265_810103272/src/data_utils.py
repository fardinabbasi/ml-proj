# data_utils.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image
import torch
from torchvision import transforms

from constants import CHAR_TO_IDX  # uses your existing mapping


# -----------------------------
# Basic I/O helpers (unchanged)
# -----------------------------
def load_image(image_path):
    """Open image as grayscale ('L')."""
    image = Image.open(image_path)
    return image.convert('L')

def load_label(json_path):
    """
    Load a label JSON and return (annotations, expression).
    If the file is missing, return ([], "").
    """
    if not json_path or not Path(json_path).exists():
        return [], ""
    with open(json_path) as f:
        data = json.load(f)
        annotations = data.get("annotations", [])
        expression = data.get("expression", "")
        return annotations, expression

# data_utils.py
from PIL import Image, ImageOps  # add ImageOps import

def crop_characters(image, bboxes, pad_ratio: float = 0.10, pad_min: int = 2, pad_max: int = 16):
    """
    Crop character regions and add a small white border so rotation/translate
    won't clip strokes. Padding is applied *after* the crop, not to the full image.
    """
    crops = []
    for box in bboxes:
        bb = box["boundingBox"]
        x, y = bb["x"], bb["y"]
        w, h = bb["width"], bb["height"]

        # tight crop
        crop = image.crop((x, y, x + w, y + h))

        # compute symmetric padding (pixels)
        pad = int(round(pad_ratio * max(w, h)))
        pad = max(pad_min, min(pad, pad_max))

        if pad > 0:
            # add white border around the crop
            crop = ImageOps.expand(crop, border=pad, fill=255)

        crops.append(crop)
    return crops

def get_x(bbox):
    """Sort key: left-to-right by x."""
    return bbox["boundingBox"]["x"]


# -----------------------------------------
# Path robustness (no changes to notebook)
# -----------------------------------------
def _safe_listdir(p: str | Path) -> List[str]:
    try:
        return os.listdir(p)
    except FileNotFoundError:
        return []

def _find_dataset_root() -> Optional[Path]:
    """
    Try to locate a dataset root that contains:
      train/images, train/labels, valid/images, valid/labels
    Sources: DATA_DIR, PROJECT_ROOT/dataset, near this file, Google Drive.
    """
    # 1) Env: DATA_DIR
    dd = os.environ.get("DATA_DIR")
    if dd:
        base = Path(dd).resolve()
        if (base / "train" / "images").exists() and (base / "train" / "labels").exists():
            return base

    # 2) Env: PROJECT_ROOT/dataset
    pr = os.environ.get("PROJECT_ROOT")
    if pr:
        cand = (Path(pr) / "dataset").resolve()
        if (cand / "train" / "images").exists() and (cand / "train" / "labels").exists():
            return cand

    # 3) Walk up from this file
    here = Path(__file__).resolve()
    for up in [here.parent, *here.parents]:
        cand = up / "dataset"
        if (cand / "train" / "images").exists() and (cand / "train" / "labels").exists():
            return cand

    # 4) Google Drive (Colab)
    drive = Path("/content/drive/MyDrive")
    if drive.exists():
        for cand in drive.glob("**/dataset"):
            if (cand / "train" / "images").exists() and (cand / "train" / "labels").exists():
                return cand.resolve()

    return None

def _fix_split_tokens(p: Path) -> Path:
    """
    Common fix: replace '/val/' with '/valid/' if that's what exists.
    Does not hard-code absolute roots.
    """
    s = str(p)
    if "/val/" in s and "/valid/" not in s:
        s2 = s.replace("/val/", "/valid/")
        return Path(s2)
    return p

def _resolve_dir(folder: str | Path, expect: str) -> Path:
    """
    Ensure 'folder' exists. If not:
      - try replacing 'val' with 'valid'
      - remap /content/dataset/... onto a discovered dataset root
      - as last resort, return the first matching split under the dataset root
    'expect' is 'images' or 'labels'.
    """
    p = Path(folder)
    if p.exists():
        return p

    # try 'val' -> 'valid'
    p2 = _fix_split_tokens(p)
    if p2.exists():
        print(f"[data_utils] Using corrected path: {p2}")
        return p2

    root = _find_dataset_root()
    if root is None:
        # give up; let caller handle empties
        return p

    # attempt to map the tail tokens onto the discovered root
    # e.g., .../dataset/val/images  -> root/valid/images (if valid exists)
    #       .../dataset/train/images -> root/train/images
    parts = [pp for pp in p.parts if pp]  # original tokens
    # guess split by scanning tokens
    split_guess = None
    for tok in ("train_aug", "train", "valid", "val", "test", "pseudo_labels"):
        if tok in parts:
            split_guess = tok
            break
    # normalize 'val' -> 'valid'
    if split_guess == "val":
        split_guess = "valid"
    if split_guess is None:
        # default to 'train' if no clue
        split_guess = "train"

    candidates = []
    if split_guess in ("train", "valid", "test"):
        candidates.append(root / split_guess / expect)
    elif split_guess == "train_aug":
        candidates.append(root / "train_aug" / expect)
    elif split_guess == "pseudo_labels":
        candidates.append(root / "pseudo_labels")
    else:
        # fallbacks
        candidates.extend([
            root / "train" / expect,
            root / "valid" / expect,
            root / "test"  / expect,
        ])

    for c in candidates:
        if c.exists():
            print(f"[data_utils] Auto-resolved '{folder}' -> {c}")
            return c

    # couldn't resolve to an existing dir; return original (will yield empties)
    return p


# ---------------------------------------------------------
# Dataset splits and preparation (your original semantics)
# ---------------------------------------------------------
# data_utils.py

def split_labeled_unlabeled(image_dir, label_dir):
    # NEW: auto-correct bad paths (val→valid, wrong root, etc.)
    image_dir = _resolve_dir(image_dir, expect="images")
    label_dir = _resolve_dir(label_dir, expect="labels")

    labeled_pairs = []
    unlabeled_pairs = []
    for img_file in _safe_listdir(image_dir):
        base_name = Path(img_file).stem
        if "-" in base_name:
            continue
        json_path = os.path.join(label_dir, f"{base_name}.json")
        img_path = os.path.join(image_dir, img_file)
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            expr = data.get("expression", "")
            if isinstance(expr, str) and expr.strip():
                labeled_pairs.append((img_path, json_path))
            else:
                unlabeled_pairs.append((img_path, json_path))
        else:
            unlabeled_pairs.append((img_path, None))
    return labeled_pairs, unlabeled_pairs


def split_labeled_unlabeled_with_pseudo(image_dir, real_label_dir, pseudo_label_dir):
    # NEW: auto-correct all three dirs
    image_dir        = _resolve_dir(image_dir,        expect="images")
    real_label_dir   = _resolve_dir(real_label_dir,   expect="labels")
    # pseudo labels dir may not exist yet — don't force it
    pseudo_label_dir = Path(_resolve_dir(pseudo_label_dir, expect="labels")) if isinstance(pseudo_label_dir, (str, Path)) else Path(pseudo_label_dir)
    # If it doesn't exist, treat as empty
    pseudo_label_dir_exists = pseudo_label_dir.exists()

    labeled_pairs = []
    unlabeled_pairs = []

    for img_file in _safe_listdir(image_dir):
        base_name = Path(img_file).stem
        if "-" in base_name:
            continue
        img_path = os.path.join(image_dir, img_file)

        json_real   = os.path.join(real_label_dir,   f"{base_name}.json")
        json_pseudo = os.path.join(pseudo_label_dir, f"{base_name}.json") if pseudo_label_dir_exists else None

        found = False
        for jp in (json_real, json_pseudo):
            if jp and os.path.exists(jp):
                with open(jp) as f:
                    data = json.load(f)
                expr = data.get("expression", "")
                labeled = (isinstance(expr, str) and expr.strip()) or (isinstance(expr, list) and any(bool(x) for x in expr))
                if labeled:
                    labeled_pairs.append((img_path, jp))
                    found = True
                    break
        if not found:
            unlabeled_pairs.append((img_path, None))
    return labeled_pairs, unlabeled_pairs



def prepare_data(
    image_dir: str | Path,
    label_dir: str | Path,
    labeled: bool = True,
    pseudo_label_dir: Optional[str | Path] = None,
    transform=None,
):
    """
    Build tensors from character crops. Preserves your original behavior:
      - sort bboxes left->right
      - for labeled=True: pair each crop with its char from 'expression';
        skip chars not in CHAR_TO_IDX
      - for labeled=False: return only images
    Robustness:
      - auto-resolves common path issues (val→valid, wrong root)
      - returns empty tensors if directories are truly missing/empty
    """
    # Resolve / correct directories if needed
    image_dir = _resolve_dir(image_dir, expect="images")
    label_dir = _resolve_dir(label_dir, expect="labels")

    images: List[torch.Tensor] = []
    labels: List[int] = []

    # If images dir still doesn't exist, return empties (non-crashing)
    if not Path(image_dir).exists():
        if labeled:
            return torch.empty(0, 1, 224, 224), torch.empty(0, dtype=torch.long)
        else:
            return torch.empty(0, 1, 224, 224)

    # Select pairs
    if labeled:
        pairs, _ = split_labeled_unlabeled(str(image_dir), str(label_dir))
    else:
        if pseudo_label_dir is None:
            # unlabeled = images without a real label JSON
            pairs = []
            for img_file in _safe_listdir(image_dir):
                base = Path(img_file).stem
                if "-" in base:
                    continue
                json_real = os.path.join(label_dir, f"{base}.json")
                if not os.path.exists(json_real):
                    pairs.append((os.path.join(image_dir, img_file), None))
        else:
            _, unlabeled_pairs = split_labeled_unlabeled_with_pseudo(
                str(image_dir), str(label_dir), str(pseudo_label_dir)
            )
            pairs = unlabeled_pairs

    # Build tensors
    for img_path, json_path in pairs:
        img = load_image(img_path)
        bboxes, expression = load_label(json_path)
        bboxes = sorted(bboxes, key=get_x)
        char_images = crop_characters(img, bboxes)

        if labeled:
            # expression may be string or list
            if isinstance(expression, str):
                expr_iter = list(expression)
            else:
                expr_iter = [c if isinstance(c, str) else "" for c in (expression or [])]

            for char_img, ch in zip(char_images, expr_iter):
                if ch not in CHAR_TO_IDX:
                    continue
                images.append(transform(char_img) if transform else transforms.ToTensor()(char_img))
                labels.append(CHAR_TO_IDX[ch])
        else:
            for char_img in char_images:
                images.append(transform(char_img) if transform else transforms.ToTensor()(char_img))

    X = torch.stack(images) if images else torch.empty(0, 1, 224, 224)
    if labeled:
        y = torch.tensor(labels, dtype=torch.long) if labels else torch.empty(0, dtype=torch.long)
        return X, y
    else:
        return X
