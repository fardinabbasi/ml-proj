# src/test_infer.py
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from constants import DEVICE, NUM_CLASSES, IDX_TO_CHAR, SAVE_PATH
from transforms_setup import val_transform
from eval_utils import predict_char
from data_utils import _resolve_dir

# optional helper if you have it
try:
    from evaluate import load_best_model  # loads best checkpoint at SAVE_PATH
except Exception:
    load_best_model = None


from pathlib import Path
import os, pandas as pd, torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from constants import DEVICE, NUM_CLASSES, IDX_TO_CHAR, SAVE_PATH, RESULTS_DIR
from transforms_setup import val_transform
from eval_utils import predict_char
from data_utils import _resolve_dir

try:
    from evaluate import load_best_model
except Exception:
    load_best_model = None

def _load_model_if_needed(model):
    if model is not None:
        return model.eval()
    if load_best_model is not None:
        return load_best_model()
    from model_def import build_model
    m = build_model(num_classes=NUM_CLASSES, in_ch=1)
    if Path(SAVE_PATH).exists():
        m.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    return m.to(DEVICE).eval()

@torch.no_grad()
def classify_expressions_from_csv(
    bbox_csv_path,
    img_dir,
    model=None,
    output_csv_path=None,
    transform=val_transform,
):
    stem = Path(os.getenv("CLASS_OUTPUT", "final_expressions.csv")).stem
    default_out = Path(RESULTS_DIR) / f"{stem}.csv"
    out_path = Path(output_csv_path) if output_csv_path else default_out

    img_dir = Path(_resolve_dir(img_dir, "images"))
    df = pd.read_csv(bbox_csv_path)
    df["image_id"] = df["image_id"].astype(str)

    model = _load_model_if_needed(model)

    results = []
    for image_id, group in df.groupby("image_id"):
        p_png = img_dir / f"{image_id}.png"
        p_jpg = img_dir / f"{image_id}.jpg"
        p_img = p_png if p_png.exists() else (p_jpg if p_jpg.exists() else None)
        if p_img is None:
            print(f"[warn] image not found for {image_id} in {img_dir}")
            continue

        image = Image.open(p_img).convert("L")
        group = group.sort_values("x")

        pred_chars = []
        for _, r in group.iterrows():
            x, y, w, h = map(int, (r["x"], r["y"], r["width"], r["height"]))
            crop = image.crop((x, y, x + w, y + h))
            ch, _ = predict_char(model, transform(crop).unsqueeze(0).to(DEVICE))
            pred_chars.append(ch)

        results.append({"image_id": str(image_id), "expression": "".join(pred_chars)})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results)[["image_id", "expression"]].to_csv(out_path, index=False)
    print(f"✅ Results saved to {out_path}")

@torch.no_grad()
def plot_test_grid_with_bboxes_labels(
    bbox_csv_path, img_dir, model=None, n_images=9, cols=3, transform=val_transform, fontsize=12
):
    img_dir = Path(_resolve_dir(img_dir, "images"))
    df = pd.read_csv(bbox_csv_path)
    df["image_id"] = df["image_id"].astype(str)
    uniq = df["image_id"].unique().tolist()
    num = min(n_images, len(uniq))
    rows = (num + cols - 1) // cols

    model = _load_model_if_needed(model)

    fig = plt.figure(figsize=(cols * 5, rows * 5))
    k = 1
    for image_id in uniq[:num]:
        ax = fig.add_subplot(rows, cols, k); k += 1
        p_png = img_dir / f"{image_id}.png"
        p_jpg = img_dir / f"{image_id}.jpg"
        p_img = p_png if p_png.exists() else (p_jpg if p_jpg.exists() else None)
        if p_img is None: ax.set_title(f"Missing: {image_id}"); ax.axis("off"); continue

        rgb = Image.open(p_img).convert("RGB")
        gray = Image.open(p_img).convert("L")
        ax.imshow(rgb); ax.axis("off"); ax.set_title(str(image_id))

        for _, r in df[df["image_id"] == image_id].sort_values("x").iterrows():
            x, y, w, h = map(int, (r["x"], r["y"], r["width"], r["height"]))
            crop = gray.crop((x, y, x + w, y + h))
            ch, _ = predict_char(model, transform(crop).unsqueeze(0).to(DEVICE))
            ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=2))
            ax.text(x, max(0, y - 3), ch, fontsize=fontsize, ha="left", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))
    plt.tight_layout(); plt.show()
