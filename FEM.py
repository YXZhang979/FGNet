import torch
import torch.nn as nn
import torch.nn.functional as F

class FEM(nn.Module):
    def __init__(self, in_channels, num_layers=4):

        super(FEM, self).__init__()
        self.num_layers = num_layers
        self.channel = 512

        self.conv1 = nn.Conv2d(in_channels, self.channel, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(self.channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(self.channel)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(self.channel)
        self.relu4 = nn.ReLU(inplace=True)


    def forward(self, x):

        x1 = self.bn1(self.relu1(self.conv1(x)))
        y1 = x1
        x2 = self.bn2(self.relu2(self.conv2(x1)))
        y2 = F.interpolate(y1, size=x2.shape[-2:], mode='bilinear', align_corners=True) + x2
        x3 = self.bn3(self.relu3(self.conv3(x2)))
        y3 = F.interpolate(y2, size=x3.shape[-2:], mode='bilinear', align_corners=True) + x3
        x4 = self.bn4(self.relu4(self.conv4(x3)))
        y4 = F.interpolate(y3, size=x4.shape[-2:], mode='bilinear', align_corners=True) + x4

        aggregated_feature = F.interpolate(y1, size=x.shape[-2:], mode='bilinear', align_corners=True) + F.interpolate(y2, size=x.shape[-2:], mode='bilinear', align_corners=True) + F.interpolate(y3, size=x.shape[-2:], mode='bilinear', align_corners=True) + F.interpolate(y4, size=x.shape[-2:], mode='bilinear', align_corners=True)
        return aggregated_feature


