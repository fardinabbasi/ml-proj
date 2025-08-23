import cv2
import math
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from . import config
from PIL import Image
from tqdm import tqdm
from torchvision.ops import nms
from .dataset import get_transform
from .model import ResNet50FPN, CustomFasterRCNN

# Visualization function for test images
def visualize_localization_results():

    df = pd.read_csv(config.OUTPUT_CSV_PATH)
    image_ids = df['image_id'].unique().tolist()

    num_samples = min(9, len(image_ids))
    selected_ids = random.sample(image_ids, num_samples)

    plt.figure(figsize=(27, 3 * math.ceil(num_samples / 3)))

    for i, image_id in enumerate(selected_ids):
        image_id = int(image_id)
        image_path = config.TEST_IMAGE_DIR / f"{image_id}.png"

        image = cv2.imread(str(image_path))
        
        if image is not None:
            bboxes = df[df['image_id'] == image_id]
            for _, row in bboxes.iterrows():
                x, y, w, h = int(row['x']), int(row['y']), int(row['width']), int(row['height'])
                
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            plt.subplot(math.ceil(num_samples / 3), 3, i + 1)
            plt.imshow(image_rgb)
            plt.title(f"Image ID: {image_id}")
            plt.axis('off')
        else:
            print(f"Image not found.")
    plt.tight_layout()
    plt.show()

# Main prediction function for generating the CSV
def generate_submission_csv():
    """
    Loads the best model and generates predictions for the entire test set,
    saving the results to a CSV file.
    """
    if not config.BEST_MODEL_PATH.exists():
        print(f"Error: Model file not found at {config.BEST_MODEL_PATH}")
        print("Please run train.py first to train and save a model.")
        return

    # Load Model
    print("Loading the best trained model...")
    custom_backbone = ResNet50FPN()
    model = CustomFasterRCNN(backbone=custom_backbone, num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    print("Model loaded successfully.")

    # Transform
    transform = get_transform(train=False)

    # Loop Over Test Images
    results = []
    test_image_paths = sorted(config.TEST_IMAGE_DIR.glob("*.png"))
    print(f"Found {len(test_image_paths)} images in the test folder.")

    with torch.no_grad():
        for img_path in tqdm(test_image_paths, desc="Generating predictions"):
            # pil_image = Image.open(img_path).convert("RGB")
            image_pil = Image.open(img_path)

            image_np = np.array(image_pil)

            if len(image_np.shape) == 3:
                gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image_np

            _ , binary_image = cv2.threshold(
                gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            image_rgb_binarized = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
            
            image = Image.fromarray(image_rgb_binarized)

            img_tensor, _ = transform(image, None)

            prediction = model([img_tensor.to(config.DEVICE)])[0]
            boxes, scores = prediction["boxes"].cpu(), prediction["scores"].cpu()

            # Filter by score threshold
            keep = scores > config.SCORE_THRESHOLD
            boxes, scores = boxes[keep], scores[keep]

            # Apply NMS
            keep_indices = nms(boxes, scores, config.NMS_IOU_THRESHOLD)
            final_boxes = boxes[keep_indices]

            # Format results for CSV
            for box in final_boxes:
                xmin, ymin, xmax, ymax = box.tolist()
                results.append({
                    "image_id": int(img_path.stem),
                    "x": xmin,
                    "y": ymin,
                    "width": xmax - xmin,
                    "height": ymax - ymin
                })

    # Save CSV
    if results:
        df = pd.DataFrame(results, columns=["image_id", "x", "y", "width", "height"])
        df.to_csv(config.OUTPUT_CSV_PATH, index=False)
        print(f"CSV saved to {config.OUTPUT_CSV_PATH}")
    else:
        print("No objects detected with the given thresholds. CSV not created.")

if __name__ == '__main__':
    generate_submission_csv()