import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as T

from . import config
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset

# Preprocessing Function
def preprocess_character_image(image, size=(32, 32)):
    """
    Binarizes, pads to square, resizes, and normalizes a single character image.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binarized_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    h, w = binarized_img.shape
    max_dim = max(w, h)
    square_canvas = np.zeros((max_dim, max_dim), dtype=np.uint8)
    
    x_pos, y_pos = (max_dim - w) // 2, (max_dim - h) // 2
    square_canvas[y_pos:y_pos + h, x_pos:x_pos + w] = binarized_img
    
    resized = cv2.resize(square_canvas, size, interpolation=cv2.INTER_AREA)

    return resized / 255.0

# Main Data Loading Function (for training and validation data)
def load_and_process_data():
    """
    Loads raw data and preprocesses character crops for training and validation.
    """
    print("Starting data loading and preprocessing...")
    labeled_train_data, unlabeled_train_data, validation_data = [], [], []

    for split in ['train', 'valid']:
        print(f"\nProcessing '{split}' split...")
        images_dir = config.DATASET_PATH / split / 'images'
        labels_dir = config.DATASET_PATH / split / 'labels'

        current_ignore_list = config.TRAIN_IGNORE if split == 'train' else config.VAL_IGNORE

        for filename in tqdm(os.listdir(images_dir)):
            base_name, _ = os.path.splitext(filename)
            if base_name in current_ignore_list:
                continue
            
            json_path = labels_dir / f"{base_name}.json"
            img_path = images_dir / filename
            if not json_path.exists():
                continue

            main_image = cv2.imread(str(img_path))
            if main_image is None:
                print(f"  [ERROR] Could not read image file: {filename}. Skipping.")
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)

            expression = annotations.get('expression')
            bboxes_data = annotations['annotations']
            bboxes = [[int(b['boundingBox']['x']), int(b['boundingBox']['y']), int(b['boundingBox']['width']), int(b['boundingBox']['height'])] for b in bboxes_data]
            
            def crop_and_process(bbox):
                x, y, w, h = bbox
                if w <= 0 or h <= 0: return None
                char_img = main_image[y:y+h, x:x+w]
                if char_img.size == 0: return None
                # Use the standalone preprocessing function
                return preprocess_character_image(char_img, config.IMG_SIZE)

            if expression and len(expression) == len(bboxes):
                for bbox, char_label in zip(bboxes, expression):
                    processed_img = crop_and_process(bbox)
                    if processed_img is not None:
                        target_list = labeled_train_data if split == 'train' else validation_data
                        target_list.append((processed_img, char_label))
            elif split == 'train' and not expression:
                for bbox in bboxes:
                    processed_img = crop_and_process(bbox)
                    if processed_img is not None:
                        unlabeled_train_data.append(processed_img)

    print("\nFinished processing all data successfully.")
    return labeled_train_data, unlabeled_train_data, validation_data

# PyTorch Dataset Classes
class UnlabeledCharacterDataset(Dataset):
    """
    Dataset for unlabeled data that returns an image and its original index.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image_tensor = torch.from_numpy(np.expand_dims(image, axis=0)).float()
        return image_tensor, idx
    

# for mean teacher method
class MeanTeacherUnlabeledDataset(Dataset):
    """
    Dataset for mean teacher unlabeled data that returns
    both version of image and its original index.
    """
    def __init__(self, data, weak_transform, strong_transform):
        self.data = data
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        img_weak = self.weak_transform(image)
        img_strong = self.strong_transform(image)
        return img_weak, img_strong, idx


class CharacterDataset(Dataset):
    """
    Dataset for labeled character data. Converts character labels to indices.
    """
    def __init__(self, data, char_map, transform=None): 
        self.data = data
        self.char_map = char_map
        self.transform = transform 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        label_idx = self.char_map[label]
        
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(np.expand_dims(image, axis=0)).float()
            
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return image_tensor, label_tensor



class PreIndexedDataset(Dataset):
    """
    A Dataset for data where labels are already integer indices.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label_idx = self.data[idx]
        image_tensor = torch.from_numpy(np.expand_dims(image, axis=0)).float()
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return image_tensor, label_tensor


class TestCharacterDataset(Dataset):
    """
    Dataset for loading character crops from test images, based on a CSV file.
    """
    def __init__(self, csv_path, images_dir):
        self.df = pd.read_csv(csv_path)
        self.images_dir = Path(images_dir)
        self.loaded_images = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = int(row['image_id'])
        
        if image_id not in self.loaded_images:
            img_path = self.images_dir / f"{image_id}.png"
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Failed to load image {img_path}. Skipping this bounding box.")
                return torch.zeros((1, *config.IMG_SIZE)), torch.tensor(image_id), torch.tensor(-1.0)
            self.loaded_images[image_id] = image

        full_image = self.loaded_images[image_id]
        
        x, y, w, h = int(row['x']), int(row['y']), int(row['width']), int(row['height'])
        
        if w <= 0 or h <= 0:
            return torch.zeros((1, *config.IMG_SIZE)), torch.tensor(image_id), torch.tensor(-1.0)

        char_image = full_image[y:y+h, x:x+w]
        
        if char_image.size == 0:
            return torch.zeros((1, *config.IMG_SIZE)), torch.tensor(image_id), torch.tensor(-1.0)
        
        processed_img_np = preprocess_character_image(char_image, config.IMG_SIZE)
        image_tensor = torch.from_numpy(np.expand_dims(processed_img_np, axis=0)).float()
        
        return image_tensor, row['image_id'], row['x']