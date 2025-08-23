# augment.py
from __future__ import annotations
from pathlib import Path
import os, json, shutil
from typing import Optional, Dict, Any, List
from torchvision import transforms
from data_utils import load_image
from transforms_setup import train_transform
from data_utils import _resolve_dir

def _has_expression(d: Dict[str, Any]) -> bool:
    expr = d.get("expression", "")
    if isinstance(expr, str):
        return bool(expr.strip())
    if isinstance(expr, list):
        return any(bool(x) for x in expr)
    return False

def _dataset_root_from_env() -> Optional[Path]:
    # Try DATA_DIR directly
    dd = os.environ.get("DATA_DIR")
    if dd:
        p = Path(dd).resolve()
        if (p / "train" / "images").exists() and (p / "train" / "labels").exists():
            return p
    # Try PROJECT_ROOT/dataset
    pr = os.environ.get("PROJECT_ROOT")
    if pr:
        p = (Path(pr) / "dataset").resolve()
        if (p / "train" / "images").exists() and (p / "train" / "labels").exists():
            return p
    return None

def _dataset_root_nearby() -> Optional[Path]:
    # Walk up from this file to find a sibling "dataset" with train/images+labels
    here = Path(__file__).resolve()
    for up in [here.parent, *here.parents]:
        cand = up / "dataset"
        if (cand / "train" / "images").exists() and (cand / "train" / "labels").exists():
            return cand.resolve()
    # Common Colab Drive base, but not hard-coding project name
    base = Path("/content/drive/MyDrive")
    if base.exists():
        for cand in base.glob("**/dataset"):
            if (cand / "train" / "images").exists() and (cand / "train" / "labels").exists():
                return cand.resolve()
    return None

def _resolve_or_fallback(p: Path, kind: str) -> Path:
    """
    If 'p' exists, return it. Otherwise, infer a dataset root and map 'kind'
    to the correct subfolder. Creates aug output dirs on demand.
    """
    if p.exists():
        return p

    root = _dataset_root_from_env() or _dataset_root_nearby()
    if root is None:
        raise FileNotFoundError(
            f"[augment] '{kind}' not found at {p} and no dataset root could be inferred.\n"
            f"Set environment variable PROJECT_ROOT or DATA_DIR, or ensure you pass existing paths."
        )

    mapping = {
        "train_images": root / "train" / "images",
        "train_labels": root / "train" / "labels",
        "aug_images":   root / "train_aug" / "images",
        "aug_labels":   root / "train_aug" / "labels",
    }
    q = mapping.get(kind, p)

    # Create aug output dirs when needed
    if kind in ("aug_images", "aug_labels"):
        q.mkdir(parents=True, exist_ok=True)

    print(f"[augment] '{kind}' not found at {p}. Using {q}")
    return q

def build_augmented_from_labeled(train_images_dir, train_labels_dir, out_aug_images_dir, out_aug_labels_dir, variants=10):
    train_images_dir   = _resolve_dir(train_images_dir,   "images")
    train_labels_dir   = _resolve_dir(train_labels_dir,   "labels")
    out_aug_images_dir = _resolve_dir(out_aug_images_dir, "images")
    out_aug_labels_dir = _resolve_dir(out_aug_labels_dir, "labels")

    out_aug_images_dir.mkdir(parents=True, exist_ok=True)
    out_aug_labels_dir.mkdir(parents=True, exist_ok=True)

    labeled_samples = []
    for json_file in os.listdir(train_labels_dir):
        if not json_file.endswith(".json"): continue
        base = Path(json_file).stem
        jp = Path(train_labels_dir) / json_file
        ip = Path(train_images_dir) / f"{base}.png"
        if not ip.exists(): continue
        data = json.load(open(jp))
        expr = data.get("expression", "")
        if (isinstance(expr, str) and expr.strip()) or (isinstance(expr, list) and any(bool(x) for x in expr)):
            labeled_samples.append((ip, jp, base))

    for ip, jp, base in labeled_samples:
        orig = load_image(ip)
        for i in range(1, variants+1):
            t = train_transform(orig)
            aug_pil = transforms.ToPILImage()(t.squeeze(0))
            aug_pil.save(out_aug_images_dir / f"{base}-{i}.png")
            json.dump(json.load(open(jp)), open(out_aug_labels_dir / f"{base}-{i}.json", "w"))

    print(f"✅ Created {variants} variants for each of {len(labeled_samples)} labeled images\n" )