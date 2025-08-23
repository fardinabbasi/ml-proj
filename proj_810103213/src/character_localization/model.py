import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torchvision.models.detection import rpn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform

# Bottleneck Block for ResNet
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

# Helper function to create layers for ResNet
def make_layer(in_channels, out_channels, blocks, stride):
    downsample = None
    if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * Bottleneck.expansion),
        )

    layers = [Bottleneck(in_channels, out_channels, stride, downsample)]
    in_channels = out_channels * Bottleneck.expansion

    for _ in range(1, blocks):
        layers.append(Bottleneck(in_channels, out_channels))

    return nn.Sequential(*layers)

# Custom ResNet50 + FPN Backbone
class ResNet50FPN(nn.Module):
    def __init__(self):
        super().__init__()

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = make_layer(256, 128, blocks=4, stride=2)
        self.layer3 = make_layer(512, 256, blocks=6, stride=2)
        self.layer4 = make_layer(1024, 512, blocks=3, stride=2)

        # FPN lateral and output convs
        in_channels_list = [512, 1024, 2048]
        self.out_channels = 256
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            l_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
            o_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.output_convs.append(o_conv)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = [x2, x3, x4]
        results = []

        last_inner = self.lateral_convs[-1](features[-1])
        results.insert(0, self.output_convs[-1](last_inner))

        for idx in range(len(features) - 2, -1, -1):
            lateral = self.lateral_convs[idx](features[idx])
            inner_top_down = nn.functional.interpolate(last_inner, size=lateral.shape[-2:], mode="nearest")
            last_inner = lateral + inner_top_down
            results.insert(0, self.output_convs[idx](last_inner))

        out = OrderedDict()
        for i, res in enumerate(results):
            out[str(i)] = res

        return out

# class TwoMLPHeadWithDropout(nn.Module):
#     def __init__(self, in_channels, representation_size, dropout_prob=0.5):
#         super().__init__()
#         self.fc6 = nn.Linear(in_channels, representation_size)
#         self.dropout1 = nn.Dropout(p=dropout_prob)
#         self.fc7 = nn.Linear(representation_size, representation_size)
#         self.dropout2 = nn.Dropout(p=dropout_prob)

#     def forward(self, x):
#         x = x.flatten(start_dim=1)
#         x = F.relu(self.fc6(x))
#         x = self.dropout1(x)
#         x = F.relu(self.fc7(x))
#         x = self.dropout2(x)
#         return x
    
# Custom Faster R-CNN Model
class CustomFasterRCNN(torch.nn.Module):
    def __init__(self, num_classes, backbone):
        super().__init__()

        self.backbone = backbone

        image_mean=[0.7874, 0.7785, 0.7747]
        image_std=[0.1023, 0.1013, 0.0879]
        
        self.transform = GeneralizedRCNNTransform(
            min_size=800, max_size=1333,
            image_mean=image_mean, image_std=image_std
        )

        anchor_generator = rpn.AnchorGenerator(
            sizes=((16,), (32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 3
        )

        rpn_head = rpn.RPNHead(
            self.backbone.out_channels,
            anchor_generator.num_anchors_per_location()[0]
        )

        self.rpn = rpn.RegionProposalNetwork(
            anchor_generator=anchor_generator, head=rpn_head,
            fg_iou_thresh=0.7, bg_iou_thresh=0.3,
            batch_size_per_image=256, positive_fraction=0.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 1000},
            nms_thresh=0.7
        )

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2'],
            output_size=7, sampling_ratio=2
        )

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(
            self.backbone.out_channels * resolution ** 2,
            representation_size
        )
        # box_head = TwoMLPHeadWithDropout(
        #     in_channels=self.backbone.out_channels * resolution ** 2,
        #     representation_size=1024,
        #     dropout_prob=0.5
        # )

        box_predictor = FastRCNNPredictor(representation_size, num_classes)

        self.roi_heads = RoIHeads(
            box_roi_pool=box_roi_pool, box_head=box_head, box_predictor=box_predictor,
            fg_iou_thresh=0.5, bg_iou_thresh=0.5,
            batch_size_per_image=512, positive_fraction=0.25,
            bbox_reg_weights=None, score_thresh=0.05,
            nms_thresh=0.5, detections_per_img=100
        )

    def forward(self, images, targets=None):

        # Transform (resize + normalize)
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)

        # Backbone
        features = self.backbone(images.tensors)

        # RPN
        proposals, proposal_losses = self.rpn(images, features, targets)

        # RoI Heads
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # Post-process
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        if self.training:
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return detections