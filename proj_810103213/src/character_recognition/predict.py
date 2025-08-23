import cv2
import torch
import pandas as pd

from . import config
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from .datasets import TestCharacterDataset



def generate_submission_file(model, char_to_idx, localization_csv_path= config.LOCALIZATION_PATH):
    """
    Predict Expression on test data.
    """
    device = config.DEVICE
    model.to(device)
    model.eval()
    
    idx_to_char = {i: char for char, i in char_to_idx.items()}

    print("Creating test dataloader from localization CSV...")
    test_dataset = TestCharacterDataset(
        csv_path=localization_csv_path,
        images_dir=config.TEST_IMAGE_DIR
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print("DataLoader created.")

    all_predictions = []
    print("\nPredicting characters in test images...")
    with torch.no_grad():
        for images, image_ids, x_coords in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            _, predicted_indices = torch.max(outputs, 1)
            
            for i in range(len(images)):
                if x_coords[i].item() == -1.0:
                    continue
                
                all_predictions.append({
                    'image_id': image_ids[i].item(),
                    'x_coord': x_coords[i].item(),
                    'char': idx_to_char.get(predicted_indices[i].item(), '')
                })

    print("Prediction complete.")

    # Convert predictions to a pandas DataFrame
    pred_df = pd.DataFrame(all_predictions)

    submission_df = pred_df.sort_values('x_coord').groupby('image_id')['char'].apply(''.join).reset_index()
    submission_df = submission_df.rename(columns={'char': 'expression'})

    # Save the final CSV
    output_path = config.SUBMISSION_FILE_PATH
    submission_df.to_csv(output_path, index=False)

    print(f"\nSubmission file created successfully at: {output_path}")
    
    return submission_df


def visualize_text_examples():
    """
    Visualize Expression on test data.
    """
    # choose 8 sample to visulize
    submission_df = pd.read_csv(config.SUBMISSION_FILE_PATH)
    sample_df = submission_df.sample(8)

    plt.figure(figsize=(16, 8))

    for i, (_, row) in enumerate(sample_df.iterrows()):
        image_id = int(row['image_id'])
        predicted_expression = row['expression']
        
        image_path = config.TEST_IMAGE_DIR / f"{image_id}.png"
        
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.subplot(2, 4, i + 1)
        plt.imshow(image_rgb)

        plt.title(f"ID: {image_id}\n{predicted_expression}")
        plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()