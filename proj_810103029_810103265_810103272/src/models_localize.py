#model_localize.py
import torch
import torch.nn as nn
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.ops import RoIAlign


class BackboneCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.features = nn.Sequential(
      nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
      nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
      nn.MaxPool2d(2),
      nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
      nn.MaxPool2d(2),
      nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
    )
    self.out_channels = 256

  def forward(self, x):
    return self.features(x)



def create_rpn(backbone_out_channels):
    anchor_generator = AnchorGenerator(
        sizes=((8, 12, 16, 24, 32, 48, 64, 96),),
        aspect_ratios=((0.15, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),)  
    )
    rpn_head = RPNHead(backbone_out_channels, anchor_generator.num_anchors_per_location()[0])
    return RegionProposalNetwork(
        anchor_generator=anchor_generator,
        head=rpn_head,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.3,
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_top_n={"training": 6000, "testing": 3000},
        post_nms_top_n={"training": 2000, "testing": 1000},
        nms_thresh=0.7
    )


class RoIHead(nn.Module):
  def __init__(self, in_channels, num_classes=2, roi_size=7, spatial_scale=1/8):
    super().__init__()
    self.roi_align = RoIAlign(output_size=(roi_size, roi_size), spatial_scale=spatial_scale, sampling_ratio=-1)
    self.fc1 = nn.Linear(in_channels * roi_size * roi_size, 1024)
    self.fc2 = nn.Linear(1024, 1024)
    self.cls_score = nn.Linear(1024, num_classes)
    self.bbox_pred = nn.Linear(1024, num_classes * 4)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, features, proposals, image_shapes):
    device = features.device
    roi_batch = []
    idxs = []
    for i, props in enumerate(proposals):
        N = props.shape[0]
        if N == 0:
            continue
        idx = torch.full((N, 1), i, dtype=props.dtype, device=device)
        roi_batch.append(torch.cat([idx, props], dim=1))
    if len(roi_batch) == 0:
        return None, None
    rois = torch.cat(roi_batch, dim=0)
    roi_feats = self.roi_align(features, rois)
    roi_feats = roi_feats.flatten(start_dim=1)
    x = self.relu(self.fc1(roi_feats))
    x = self.relu(self.fc2(x))
    cls_logits = self.cls_score(x)
    bbox_deltas = self.bbox_pred(x)
    return cls_logits, bbox_deltas


