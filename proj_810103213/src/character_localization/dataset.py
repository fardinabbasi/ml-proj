import cv2
import json
import torch
import numpy as np
import torchvision.transforms.v2 as T

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class CharacterLocalizationDataset(Dataset):
    """
    A Dataset Class to load images and their corresponding bounding box annotations.
    """
    def __init__(self, image_dir, label_dir, transforms=None, ignore_list=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms

        # All image files
        all_files = sorted([p for p in self.image_dir.glob('*.png')])

        # Filter ignore list 
        if ignore_list:
            self.image_files = [
                f for f in all_files if f.stem not in ignore_list
            ]
            print(f"Original number of files: {len(all_files)}. After ignoring: {len(self.image_files)} files.")
        else:
            self.image_files = all_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # image = Image.open(img_path).convert("RGB")
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
        label_path = self.label_dir / f"{img_path.stem}.json"

        boxes = []
        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for ann_obj in data['annotations']:
                bbox = ann_obj['boundingBox']
                xmin = bbox['x']
                ymin = bbox['y']
                width = bbox['width']
                height = bbox['height']
                xmax = xmin + width
                ymax = ymin + height
                boxes.append([xmin, ymin, xmax, ymax])

        # Convert into torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        num_objs = len(boxes)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

def get_transform(train):
    """
    Defines the image transformations.
    """
    transforms = []
    # if train:
       
        # Apply rotation and scaling.
        # and scale it to a random size between 85% and 115% of the original.
        # transforms.append(T.RandomAffine(degrees=(-5, 5), scale=(0.95, 1.05)))
        
        # transforms.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    return T.Compose(transforms)

def collate_fn(batch):
    """
    Custom collate function for the DataLoader.
    """
    return tuple(zip(*batch))