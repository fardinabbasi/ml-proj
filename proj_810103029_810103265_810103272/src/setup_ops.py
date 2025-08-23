# setup_ops.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Optional
from PIL import Image, ExifTags, ImageOps
import os, shutil
from data_utils import _resolve_dir  # reuse existing resolver


# --- utilities ---------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _img_label_paths(root_img: Path, root_lbl: Path, file_id: str) -> Tuple[Path, Path]:
    return root_img / f"{file_id}.png", root_lbl / f"{file_id}.json"

# --- 1) fix orientation (safe no-ops for PNGs without EXIF) ------------------

def fix_orientation_and_save(image_path: Path) -> None:
    try:
        img = Image.open(image_path)
        exif = getattr(img, "_getexif", lambda: None)()
        if exif:
            orientation_key = next((k for k,v in ExifTags.TAGS.items() if v == "Orientation"), None)
            if orientation_key and orientation_key in exif:
                o = exif[orientation_key]
                if o == 3:   img = img.rotate(180, expand=True)
                elif o == 6: img = img.rotate(270, expand=True)
                elif o == 8: img = img.rotate(90,  expand=True)
        img.save(image_path)
    except Exception as e:
        print(f"[setup] skip orientation for {image_path.name}: {e}")

def bulk_fix_orientations(folders: Iterable[Path]) -> None:
    for folder in folders:
        folder = Path(folder)
        if not folder.exists(): continue
        for f in folder.iterdir():
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                fix_orientation_and_save(f)
    print("[setup] orientation pass done.")

# --- 2) remove bad IDs (images + labels) -------------------------------------

def remove_ids(image_dir: Path, label_dir: Path, ids: Iterable[str]) -> Tuple[int,int]:
    removed_img = removed_lbl = 0
    for file_id in ids:
        img_p, lbl_p = _img_label_paths(Path(image_dir), Path(label_dir), file_id)
        if img_p.exists():
            img_p.unlink(); removed_img += 1; print(f"[setup] deleted image: {img_p.name}")
        if lbl_p.exists():
            lbl_p.unlink(); removed_lbl += 1; print(f"[setup] deleted label: {lbl_p.name}")
    return removed_img, removed_lbl

# --- 3) sanity scan (optional) ----------------------------------------------

def scan_labels(label_dir: Path) -> None:
    import json
    label_dir = Path(label_dir)
    n, empty, non_empty = 0, 0, 0
    for jp in label_dir.glob("*.json"):
        n += 1
        try:
            d = json.load(open(jp))
            expr = d.get("expression", "")
            ok = (isinstance(expr, str) and expr.strip()) or (isinstance(expr, list) and any(bool(x) for x in expr))
            if ok: non_empty += 1
            else:  empty += 1
        except Exception as e:
            print(f"[setup] bad json {jp.name}: {e}")
    print(f"[setup] labels in {label_dir}: total={n}, non_empty={non_empty}, empty={empty}")

# --- 4) one-call convenience -------------------------------------------------

def clean_dataset(
    train_images: Path, train_labels: Path,
    valid_images: Path, valid_labels: Path,
    bad_train_ids: Optional[Iterable[str]] = None,
    bad_valid_ids: Optional[Iterable[str]] = None,
    aux_dirs: Optional[Tuple[Path, Path, Path]] = None,  # (aug_images, aug_labels, pseudo_labels)
    run_orientation_fix: bool = True
) -> None:
    """Drop known-bad files, fix EXIF rotation, and ensure aux dirs exist."""
    # --- resolve dirs first ---
    train_images = Path(_resolve_dir(train_images, expect="images"))
    train_labels = Path(_resolve_dir(train_labels, expect="labels"))
    valid_images = Path(_resolve_dir(valid_images, expect="images"))
    valid_labels = Path(_resolve_dir(valid_labels, expect="labels"))
    if aux_dirs:
        aux_dirs = tuple(
            Path(_resolve_dir(p, expect="labels" if "label" in str(p).lower() else "images"))
            for p in aux_dirs
        )

    # --- ensure aux dirs ---
    if aux_dirs:
        for p in aux_dirs:
            _ensure_dir(Path(p))

    # --- delete bad IDs ---
    if bad_train_ids:
        remove_ids(train_images, train_labels, bad_train_ids)
    if bad_valid_ids:
        remove_ids(valid_images, valid_labels, bad_valid_ids)

    # --- optional orientation fix ---
    if run_orientation_fix:
        bulk_fix_orientations([train_images, valid_images])

    # --- report ---
    scan_labels(train_labels)
    scan_labels(valid_labels)
    print("[setup] cleanup complete.")

