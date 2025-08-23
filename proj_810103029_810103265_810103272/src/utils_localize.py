# filename: utils_localize.py
from pathlib import Path
import torch
import random
import numpy as np
from PIL import Image
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def move_target_to_device(t, device):
  out = {}
  for k, v in t.items():
      out[k] = v.to(device) if torch.is_tensor(v) else v  
  return out

def apply_deltas_to_boxes(boxes, deltas):
  widths  = boxes[:, 2] - boxes[:, 0]
  heights = boxes[:, 3] - boxes[:, 1]
  ctr_x   = boxes[:, 0] + 0.5 * widths
  ctr_y   = boxes[:, 1] + 0.5 * heights
  dx, dy, dw, dh = deltas.unbind(dim=1)
  pred_ctr_x = dx * widths + ctr_x
  pred_ctr_y = dy * heights + ctr_y
  pred_w     = torch.exp(dw) * widths
  pred_h     = torch.exp(dh) * heights
  x1 = pred_ctr_x - 0.5 * pred_w
  y1 = pred_ctr_y - 0.5 * pred_h
  x2 = pred_ctr_x + 0.5 * pred_w
  y2 = pred_ctr_y + 0.5 * pred_h
  return torch.stack([x1, y1, x2, y2], dim=1)

def get_original_image_size(img_dir, img_name):
  img_path = os.path.join(img_dir, img_name)
  with Image.open(img_path) as img:
      return img.size  # (width, height)
