# filename: dataset_localize.py
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps
import os

class DataSetMatch(Dataset):
  def __init__(self, img_dir, label_dir, s, noisy_files=None, test_mode=False):
    self.img_dir = img_dir
    self.label_dir = label_dir
    all_files = os.listdir(img_dir)
    if noisy_files is not None:
        all_files = [f for f in all_files if f not in noisy_files]
    self.img_files = sorted(all_files)
    self.scal = s
    self.to_grayscale1 = transforms.Grayscale(num_output_channels=1)
    self.to_tensor = transforms.ToTensor()
    self.test_mode = test_mode

  def __len__(self):
    return len(self.img_files)

  def __getitem__(self, i):
    img_name = self.img_files[i]
    img_path = os.path.join(self.img_dir, img_name)
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')
    img = self.to_grayscale1(img)

    w, h = img.size
    new_h = self.scal
    new_w = int(w * (self.scal / h))
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    x_scale = new_w / w
    y_scale = new_h / h

    img = self.to_tensor(img)

    if self.test_mode:
        return img, {"img_name": img_name, "size": (new_h, new_w)}
    else:
        lab_path = os.path.join(
            self.label_dir,
            img_name.replace('.png', '.json')
        )
        boxes = []
        with open(lab_path, 'r') as f:
            data = json.load(f)
            for obj in data['annotations']:
                bbox = obj['boundingBox']
                x = bbox['x'] * x_scale
                y = bbox['y'] * y_scale
                bw = bbox['width'] * x_scale
                bh = bbox['height'] * y_scale
                x_min = x
                y_min = y
                x_max = x + bw
                y_max = y + bh
                boxes.append([x_min, y_min, x_max, y_max])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        return img, {"boxes": boxes, "labels": labels, "size": (new_h, new_w)}


            
import torch

def collate_fn(batch, mean: float = 0.5, std: float = 0.5):

    imgs, targets = list(zip(*batch))  
    heights = [img.shape[1] for img in imgs]
    widths  = [img.shape[2] for img in imgs]
    max_height = max(heights)
    max_width  = max(widths)

    padded_imgs = []
    for img in imgs:
        padded = torch.ones((1, max_height, max_width), dtype=img.dtype)
        padded[:, :img.shape[1], :img.shape[2]] = img
        padded_imgs.append(padded)

    imgs_tensor = torch.stack(padded_imgs)  # [B,1,H,W]
    imgs_tensor.sub_(mean).div_(std + 1e-8)

    return imgs_tensor, list(targets)



