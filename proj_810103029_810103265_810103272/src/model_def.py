import torch.nn as nn
from torchvision import models
from utils import NUM_CLASSES

def build_model(num_classes=NUM_CLASSES, in_ch=1):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
