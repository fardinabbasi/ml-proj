# self_training.py
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from constants import DEVICE, SAVE_PATH
from train import train_model
from eval_utils import (
    generate_pseudo_labels_character_level,
    evaluate_expressions,
    evaluate,
)
from data_utils import (
    prepare_data,
    _resolve_dir,
    split_labeled_unlabeled_with_pseudo,
)
from transforms_setup import val_transform


# self_training.py
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from constants import DEVICE, SAVE_PATH
from train import train_model
from eval_utils import generate_pseudo_labels_character_level, evaluate_expressions, evaluate
from data_utils import prepare_data, _resolve_dir, split_labeled_unlabeled_with_pseudo
from transforms_setup import val_transform

def _dataset_root_from_labels(label_dir_str: str) -> Path:
    return Path(label_dir_str).resolve().parent.parent  # .../train/labels -> .../dataset

def self_training_loop(
    model,
    all_image_dir,
    real_label_dir,
    pseudo_label_dir,
    val_loader,
    val_img_dir,
    val_label_dir,
    conf_thresh=0.95,
    max_per_round=50,
    epochs_per_round=15,
    max_rounds=3,
    batch_size=4,
):
    all_image_dir    = str(_resolve_dir(all_image_dir,    "images"))
    real_label_dir   = str(_resolve_dir(real_label_dir,   "labels"))
    pseudo_label_dir = str(_resolve_dir(pseudo_label_dir, "labels"))
    val_img_dir      = str(_resolve_dir(val_img_dir,      "images"))
    val_label_dir    = str(_resolve_dir(val_label_dir,    "labels"))

    data_root = _dataset_root_from_labels(real_label_dir)
    aug_images = str(_resolve_dir(data_root / "train_aug" / "images", "images"))
    aug_labels = str(_resolve_dir(data_root / "train_aug" / "labels", "labels"))

    for round_num in range(1, max_rounds + 1):
        _, unlabeled_pairs = split_labeled_unlabeled_with_pseudo(all_image_dir, real_label_dir, pseudo_label_dir)
        if len(unlabeled_pairs) == 0:
            print("[self-train] All data labeled. Stopping.")
            break

        to_pseudo = [ip for ip, _ in unlabeled_pairs[:max_per_round]]

        generate_pseudo_labels_character_level(
            model=model,
            image_list=to_pseudo,
            label_dir_out=pseudo_label_dir,
            conf_thresh=conf_thresh,
            bbox_label_dir=real_label_dir,  # <— critical, no hard-code
        )

        X_real,   y_real   = prepare_data(all_image_dir, real_label_dir,   labeled=True, transform=val_transform)
        X_aug,    y_aug    = prepare_data(aug_images,    aug_labels,       labeled=True, transform=val_transform)
        X_pseudo, y_pseudo = prepare_data(all_image_dir, pseudo_label_dir, labeled=True, transform=val_transform)

        ds = ConcatDataset([
            TensorDataset(X_real, y_real),
            TensorDataset(X_aug, y_aug),
            TensorDataset(X_pseudo, y_pseudo),
        ])
        train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        train_model(model, train_loader, val_loader, epochs=epochs_per_round)

        if os.path.exists(SAVE_PATH):
            model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        else:
            print(f"[self-train] WARN: {SAVE_PATH} not found; using in-memory model.")

        val_acc = evaluate(model, val_loader)
        avg_dist, _ = evaluate_expressions(model, val_img_dir, val_label_dir)
        print(f"[self-train] round {round_num} — val_acc={val_acc*100:.2f}%, lev={avg_dist:.3f}")

    return model
