from pathlib import Path
import os
import matplotlib.pyplot as plt

def show_saved_augmented_variants(base_id, aug_images_dir, max_variants=10, per_row=5):
    """
    Display the saved augmented images for a given base_id:
    expects files like {base_id}-1.png, {base_id}-2.png, ...
    """
    aug_images_dir = Path(aug_images_dir)
    imgs = []
    for i in range(1, max_variants + 1):
        p = aug_images_dir / f"{base_id}-{i}.png"
        if p.exists():
            imgs.append(p)
    if not imgs:
        print(f"No saved augmented images found for id '{base_id}' in {aug_images_dir}")
        return
    rows = (len(imgs) + per_row - 1) // per_row
    plt.figure(figsize=(per_row * 2.1, rows * 2.1))
    for idx, p in enumerate(imgs):
        plt.subplot(rows, per_row, idx + 1)
        plt.imshow(plt.imread(p), cmap='gray')
        plt.title(p.name, fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_augmented_from_labeled(train_labels_dir, aug_images_dir, base_id=None, max_variants=10, per_row=5):
    """
    If base_id is None, pick the first labeled id that has saved augmented variants.
    Otherwise, visualize that base_id's variants.
    """
    train_labels_dir = Path(train_labels_dir)
    aug_images_dir   = Path(aug_images_dir)

    if base_id is None:
        # find first label json that has at least one corresponding augmented image
        for jf in sorted(train_labels_dir.glob("*.json")):
            candidate = jf.stem
            if any((aug_images_dir / f"{candidate}-{i}.png").exists() for i in range(1, max_variants + 1)):
                base_id = candidate
                break
    if base_id is None:
        print("No augmented images found to visualize.")
        return

    print(f"Showing saved augmented variants for id '{base_id}'")
    show_saved_augmented_variants(base_id, aug_images_dir, max_variants=max_variants, per_row=per_row)
