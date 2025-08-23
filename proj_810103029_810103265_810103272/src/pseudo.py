# pseudo.py
import os, json
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from constants import IDX_TO_CHAR, DEVICE
from data_utils import load_image, load_label, crop_characters, get_x
from transforms_setup import val_transform

@torch.no_grad()
def generate_pseudo_labels_character_level(
    model,
    image_list,           # list[str|Path] of image paths to pseudo-label
    real_label_dir,       # str|Path  <-- where your bbox JSONs live (train_labels)
    label_dir_out,        # str|Path  <-- where to write pseudo JSONs
    conf_thresh=0.85
):
    real_label_dir = Path(real_label_dir)
    label_dir_out  = Path(label_dir_out)
    os.makedirs(label_dir_out, exist_ok=True)

    model.eval()
    count_saved = total_chars = total_confident = 0

    for img_path in tqdm(image_list, desc="Generating Char-level"):
        img_path = Path(img_path)
        base_name = img_path.stem
        if "-" in base_name:
            continue

        json_in_path  = real_label_dir / f"{base_name}.json"
        json_out_path = label_dir_out  / f"{base_name}.json"
        if not json_in_path.exists():
            continue

        image = load_image(img_path)
        bboxes, _ = load_label(json_in_path)
        if not bboxes:
            continue

        bboxes = sorted(bboxes, key=get_x)
        crops = crop_characters(image, bboxes)

        pseudo_expr = []
        for crop in crops:
            tensor = val_transform(crop).unsqueeze(0).to(DEVICE)
            probs = F.softmax(model(tensor), dim=1)
            conf, pred_idx = probs.max(dim=1)
            conf_val = float(conf.item())
            pseudo_expr.append(IDX_TO_CHAR[pred_idx.item()] if conf_val >= conf_thresh else "")
            total_confident += int(conf_val >= conf_thresh)
            total_chars += 1

        if any(c != "" for c in pseudo_expr):
            with open(json_out_path, "w") as f:
                json.dump({"annotations": bboxes, "expression": pseudo_expr}, f)
            count_saved += 1

    pct = (total_confident/total_chars*100) if total_chars else 0.0
    print(f"[pseudo] saved: {count_saved} | confident chars: {total_confident}/{total_chars} ({pct:.1f}%)")
