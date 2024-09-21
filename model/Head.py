import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, in_channels, num_classes=21):
        super(Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=7, padding=1)
        self.conv3 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        return x
