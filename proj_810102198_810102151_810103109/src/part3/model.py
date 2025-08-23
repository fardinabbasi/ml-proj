import torch.nn as nn
import torchvision.models as models
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from config import NUM_CLASSES

class ResNet18Classifier(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):
        super(ResNet18Classifier, self).__init__()
        
        self.backbone = models.resnet18(weights=None)
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)