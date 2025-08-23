

import os
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader
from constants import DEVICE, SAVE_PATH, RESULTS_DIR
from model_def import build_model
from data_utils import _resolve_dir, prepare_data
from transforms_setup import val_transform
from eval_utils import evaluate, evaluate_expressions, show_misclassified_grid_detailed, character_accuracy_report

def load_best_model(save_path: str = SAVE_PATH):
    m = build_model(in_ch=1)
    if Path(save_path).exists():
        m.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return m.to(DEVICE).eval()

def run_evaluation(model=None, val_img_dir=None, val_label_dir=None, show_grid=True, batch_size=16):
    # auto-resolve if not passed
    if val_img_dir is None or val_label_dir is None:
        data_dir = Path(os.getenv("DATA_DIR", Path.cwd() / "dataset"))
        val_img_dir   = str(_resolve_dir(data_dir / "valid" / "images", "images"))
        val_label_dir = str(_resolve_dir(data_dir / "valid" / "labels", "labels"))

    if model is None:
        model = load_best_model()

    X_val, y_val = prepare_data(val_img_dir, val_label_dir, labeled=True, transform=val_transform)
    loader_val = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False, drop_last=False)

    val_acc = evaluate(model, loader_val)
    avg_dist, misclassified = evaluate_expressions(model, val_img_dir, val_label_dir)

    print(f"\n- Final Validation Accuracy: {val_acc*100:.2f}%")
    print(f"- Final Avg Levenshtein Distance: {avg_dist:.3f}")

    if show_grid and misclassified:
        show_misclassified_grid_detailed(misclassified, max_samples=len(misclassified), cols=10)

    print("\n📊 Character-wise Accuracy Report:")
    for c, acc, correct, total in character_accuracy_report(model, loader_val):
        print(f"{c:^6} {acc*100:8.2f} {correct:^8} {total:^8}")

    return {"val_acc": val_acc, "avg_levenshtein": avg_dist}
