#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import math


def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio,
                      inp * expand_ratio,
                      3,
                      stride,
                      1,
                      groups=inp * expand_ratio,
                      bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


class PFLDInference(nn.Module):
    def __init__(self, num_keypoints=98):
        super(PFLDInference, self).__init__()
        
        # Backbone layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Inverted Residual Blocks
        self.conv3_1 = InvertedResidual(64, 64, 2, False, 2)
        self.block3_2 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_3 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_4 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_5 = InvertedResidual(64, 64, 1, True, 2)
        
        self.conv4_1 = InvertedResidual(64, 128, 2, False, 2)
        self.conv5_1 = InvertedResidual(128, 128, 1, False, 4)
        self.block5_2 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_3 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_4 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_5 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_6 = InvertedResidual(128, 128, 1, True, 4)
        
        # Heatmap Regression Head
        self.conv6_1 = InvertedResidual(128, 16, 1, False, 2)
        
        # Upsampling layers for heatmap generation
        self.heatmap_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, num_keypoints, 3, padding=1),
            nn.Sigmoid()  # Normalize to [0,1]
        )
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))       # [B,64,56,56]
        x = self.relu(self.bn2(self.conv2(x)))       # [B,64,56,56]
        
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)                     # [B,64,28,28]
        
        x = self.conv4_1(out1)                      # [B,128,14,14]
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)                        # [B,128,14,14]
        
        x = self.conv6_1(x)                         # [B,16,14,14]
        heatmap = self.heatmap_head(x)               # [B,98,56,56]
        
        return out1, heatmap


class AuxiliaryNet(nn.Module):
    def __init__(self):
        super(AuxiliaryNet, self).__init__()
        self.conv1 = conv_bn(64, 128, 3, 2)
        self.conv2 = conv_bn(128, 128, 3, 1)
        self.conv3 = conv_bn(128, 32, 3, 2)
        self.conv4 = conv_bn(32, 128, 7, 1)
        self.max_pool1 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss for Heatmap Regression
    Reference: Wang X. et al. Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression
    """
    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta

    def forward(self, pred, target):
        delta = (target - pred).abs()
        A = self.omega * (1/(1 + (self.theta/self.epsilon)**(self.alpha - target))) * \
           (self.alpha - target)*((self.theta/self.epsilon)**(self.alpha - target -1))/self.epsilon
        C = self.theta*A - self.omega*torch.log(1 + (self.theta/self.epsilon)**(self.alpha - target))
        
        # 分段计算损失
        loss = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + (delta/self.epsilon)**(self.alpha - target)),
            A * delta - C
        )
        return loss.mean()


if __name__ == '__main__':
    # 验证前向传播
    input = torch.randn(1, 3, 112, 112)
    pfld = PFLDInference()
    auxiliary = AuxiliaryNet()
    features, heatmap = pfld(input)
    angle = auxiliary(features)
    
    print("Heatmap shape:", heatmap.shape)  # 应输出 [1, 98, 56, 56]
    print("Angle shape:", angle.shape)      # 应输出 [1, 3]