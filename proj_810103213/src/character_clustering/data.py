import json
import numpy as np

from . import config
from PIL import Image
from pathlib import Path

def custom_resize(pil_img, target_size=(32, 32)):

    pil_img.thumbnail(target_size, Image.Resampling.LANCZOS)
    padded_img = Image.new("L", target_size, (0))
    paste_position = ((target_size[0] - pil_img.size[0]) // 2,
                      (target_size[1] - pil_img.size[1]) // 2)
    padded_img.paste(pil_img, paste_position)
    return padded_img

def extract_characters():

    # List of images to be ignored
    train_ignore = config.TRAIN_IGNORE

    dataset_path = config.DATASET_PATH
    results_path = config.RESULTS_DIR
    train_path = dataset_path / "train"
    image_dir = train_path / "images"
    label_dir = train_path / "labels"

    output_dir = results_path / "characters"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not label_dir.exists() or not image_dir.exists():
        print(f"No images found")
        return []

    character_images = []
    print("Loading and cropping character images...")

    for label_file in label_dir.glob("*.json"):
        # Check if the file stem is in the ignore list
        if label_file.stem in train_ignore:
            print(f"Skipping ignored file: {label_file.name}")
            continue

        # Find the corresponding image file
        image_path = image_dir / f"{label_file.stem}.png"
        if not image_path.exists():
            image_path = image_dir / f"{label_file.stem}.png"
            if not image_path.exists():
                continue

        with open(label_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        img = Image.open(image_path)

        for i, annotation in enumerate(data.get("annotations", [])):
            bbox = annotation.get("boundingBox")
            if bbox:
                box = (bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height'])
                char_img = img.crop(box)

                character_images.append(char_img)

                output_filename = f"{label_file.stem}_char_{i}.png"
                char_img.save(output_dir / output_filename)

    print(f"Found and cropped {len(character_images)} characters.")
    return character_images


def load_characters(cache_path):
  
    character_images = []
    print(f"Loading characters from directory: {cache_path}...")
    
    image_files = sorted(list(cache_path.glob("*.png"))) 
    if not image_files:
        print("Warning: This directory is empty.")
        return []
        
    for img_file in image_files:
        character_images.append(Image.open(img_file))
        
    print(f"Found and loaded {len(character_images)} characters from cache.")
    return character_images
