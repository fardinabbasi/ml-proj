# eval_utils.py
from __future__ import annotations

# --- Optional dependency: Levenshtein (auto-install if missing) --------------
try:
    import Levenshtein  # type: ignore
except Exception:
    try:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "python-Levenshtein"])
        import Levenshtein  # type: ignore
    except Exception:
        Levenshtein = None  # we'll fallback to a local edit distance

# --- Std / third-party imports ----------------------------------------------
import os, json, math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# --- Project imports (kept identical to your setup) --------------------------
from constants import DEVICE, IDX_TO_CHAR, CHARS
from transforms_setup import val_transform
from data_utils import (
    load_image, load_label, crop_characters, get_x,
    _resolve_dir,  # path auto-resolver to handle wrong roots/val→valid
)

# -----------------------------------------------------------------------------
# Utility: fallback edit distance (only used if Levenshtein not available)
# -----------------------------------------------------------------------------
def _edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[m]

# -----------------------------------------------------------------------------
# Predict one character from a prepared tensor image (kept same API)
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_char(model, tensor_img: torch.Tensor) -> Tuple[str, int]:
    model.eval()
    if tensor_img.dim() == 3:
        tensor_img = tensor_img.unsqueeze(0)
    tensor_img = tensor_img.to(DEVICE)
    logits = model(tensor_img)
    pred_idx = int(logits.argmax(dim=1).item())
    return IDX_TO_CHAR[pred_idx], pred_idx

# -----------------------------------------------------------------------------
# Overall character accuracy on a DataLoader (kept same API)
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader: DataLoader) -> float:
    model.eval()
    total_correct = 0
    total_samples = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)
    return (total_correct / total_samples) if total_samples else 0.0

# -----------------------------------------------------------------------------
# Character-wise accuracy report (kept same behavior)
# -----------------------------------------------------------------------------
@torch.no_grad()
def character_accuracy_report(model, loader: DataLoader):
    from collections import Counter
    model.eval()
    char_correct = Counter()
    char_total = Counter()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=1)
        for gt, pr in zip(y.cpu().tolist(), preds.cpu().tolist()):
            char_total[IDX_TO_CHAR[gt]] += 1
            if gt == pr:
                char_correct[IDX_TO_CHAR[gt]] += 1
    results = []
    for c in CHARS:
        total = char_total[c]
        acc = (char_correct[c] / total) if total > 0 else 0.0
        results.append((c, acc, char_correct[c], total))
    return results

# -----------------------------------------------------------------------------
# Misclassified grid viewer (kept same visuals)
# -----------------------------------------------------------------------------
def show_misclassified_grid_detailed(misclassified: List[Dict[str, Any]], max_samples: Optional[int]=None, cols: int=10):
    num_samples = min(len(misclassified), max_samples) if max_samples is not None else len(misclassified)
    if num_samples == 0:
        print("[eval] No misclassified samples to show.")
        return
    rows = math.ceil(num_samples / cols)
    plt.figure(figsize=(cols*2.2, rows*2.8))
    for idx, m in enumerate(misclassified[:num_samples]):
        plt.subplot(rows, cols, idx + 1)
        plt.text(0.5, 1.12, f"{m.get('source_img','')}", fontsize=8, ha='center', va='top', transform=plt.gca().transAxes)
        plt.text(0.5, 1.02, f"GT: {m.get('gt','?')} | P: {m.get('pred','?')}", fontsize=9, ha='center', va='top', transform=plt.gca().transAxes)
        plt.imshow(m["crop"], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Expression-level evaluation (robust to list/str & unequal lengths)
# -----------------------------------------------------------------------------
# eval_utils.py (ONLY the two functions below if you prefer)

from data_utils import _resolve_dir, load_image, load_label, crop_characters, get_x
from transforms_setup import val_transform
import torch, json, os
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
from constants import DEVICE, IDX_TO_CHAR

@torch.no_grad()
def generate_pseudo_labels_character_level(
    model,
    image_list,
    label_dir_out,
    conf_thresh=0.85,
    bbox_label_dir=None,
):
    label_dir_out = Path(label_dir_out)
    label_dir_out.mkdir(parents=True, exist_ok=True)
    if bbox_label_dir is None:
        if not image_list:
            print("[pseudo] image_list empty.")
            return
        first = Path(image_list[0])
        bbox_label_dir = first.parent.parent / "labels"
    bbox_label_dir = Path(_resolve_dir(bbox_label_dir, expect="labels"))

    model.eval()
    count_saved = 0; total_chars = 0; total_conf = 0

    for img_path in tqdm(image_list, desc="Generating Char-level"):
        base = Path(img_path).stem
        if "-" in base:  # skip augmented
            continue
        image = load_image(img_path)
        label_json = Path(bbox_label_dir) / f"{base_name}.json"
        if not label_json.exists():
            continue
        bboxes, _ = load_label(label_json)
        if not bboxes:
            continue

        bboxes = sorted(bboxes, key=get_x)
        crops = crop_characters(image, bboxes)

        pseudo = []
        for crop in crops:
            t = val_transform(crop).unsqueeze(0).to(DEVICE)
            logits = model(t)
            probs = F.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
            c = float(conf.item()); p = IDX_TO_CHAR[int(idx.item())]
            if c >= conf_thresh:
                pseudo.append(p); total_conf += 1
            else:
                pseudo.append("")
            total_chars += 1

        if any(x != "" for x in pseudo):
            (label_dir_out / f"{base}.json").write_text(
                json.dumps({"annotations": bboxes, "expression": pseudo})
            )
            count_saved += 1

    pct = (total_conf / total_chars * 100) if total_chars else 0.0
    print(f"[pseudo] saved: {count_saved} | confident chars: {total_conf}/{total_chars} ({pct:.1f}%)")

@torch.no_grad()
def evaluate_expressions(model, image_dir, label_dir):
    # resolve dirs
    image_dir = str(_resolve_dir(image_dir, expect="images"))
    label_dir = str(_resolve_dir(label_dir, expect="labels"))

    try:
        import Levenshtein
        dist_fn = Levenshtein.distance
    except Exception:
        # simple fallback
        def dist_fn(a,b):
            n, m = len(a), len(b)
            if n == 0: return m
            if m == 0: return n
            dp = list(range(m + 1))
            for i in range(1, n + 1):
                prev = dp[0]; dp[0] = i
                for j in range(1, m + 1):
                    tmp = dp[j]
                    cost = 0 if a[i-1] == b[j-1] else 1
                    dp[j] = min(dp[j]+1, dp[j-1]+1, prev+cost)
                    prev = tmp
            return dp[m]

    model.eval()
    dists = []; misclassified = []
    for img_file in os.listdir(image_dir):
        base = Path(img_file).stem
        ip = os.path.join(image_dir, img_file)
        jp = os.path.join(label_dir, base + ".json")
        if not os.path.exists(jp): continue

        img = load_image(ip)
        bboxes, expr = load_label(jp)
        if isinstance(expr, list):
            gt = "".join([(c if isinstance(c,str) and len(c)==1 else "") for c in expr])
        else:
            gt = expr or ""
        if not gt: continue

        bboxes = sorted(bboxes, key=get_x)
        crops = crop_characters(img, bboxes)

        pred = []
        for crop in crops:
            t = val_transform(crop).unsqueeze(0).to(DEVICE)
            logits = model(t)
            pred.append(IDX_TO_CHAR[int(logits.argmax(dim=1).item())])
        pred = "".join(pred)

        # collect per-char errors for positions within min length
        L = min(len(gt), len(pred))
        for i in range(L):
            if gt[i] != pred[i]:
                misclassified.append({"crop": crops[i], "gt": gt[i], "pred": pred[i], "source_img": img_file, "char_index": i})
        dists.append(dist_fn(gt, pred))

    avg = (sum(dists)/len(dists)) if dists else float("nan")
    print(f"- Avg Levenshtein Distance: {avg:.3f}" if dists else "- Avg Levenshtein Distance: NaN (no labeled expr)")
    print(f"- Misclassified characters: {len(misclassified)}")
    return avg, misclassified
