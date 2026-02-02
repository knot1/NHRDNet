#!/usr/bin/env python3
# -*- coding:utf-8 -*-

######################################################
#
# pfld.py -
# written by  zhaozhichao
#
######################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    if self.use_res_connect:
      return x + self.conv(x)
    else:
      return self.conv(x)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
  """Pad to 'same' shape outputs."""
  if d > 1:
    k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
  if p is None:
    p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
  return p

class Conv(nn.Module):
  """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
  default_act = nn.SiLU()  # default activation

  def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
    """Initialize Conv layer with given arguments including activation."""
    super().__init__()
    self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
    self.bn = nn.BatchNorm2d(c2)
    self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

  def forward(self, x):
    """Apply convolution, batch normalization and activation to input tensor."""
    return self.act(self.bn(self.conv(x)))

  def forward_fuse(self, x):
    """Perform transposed convolution of 2D data."""
    return self.act(self.conv(x))

class DCNv2(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
               padding=1, dilation=1, groups=1, deformable_groups=1):
    super(DCNv2, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = (kernel_size, kernel_size)
    self.stride = (stride, stride)
    self.padding = (padding, padding)
    self.dilation = (dilation, dilation)
    self.groups = groups
    self.deformable_groups = deformable_groups

    self.weight = nn.Parameter(
      torch.empty(out_channels, in_channels, *self.kernel_size)
    )
    self.bias = nn.Parameter(torch.empty(out_channels))

    out_channels_offset_mask = (self.deformable_groups * 3 *
                                self.kernel_size[0] * self.kernel_size[1])
    self.conv_offset_mask = nn.Conv2d(
      self.in_channels,
      out_channels_offset_mask,
      kernel_size=self.kernel_size,
      stride=self.stride,
      padding=self.padding,
      bias=True,
    )
    self.bn = nn.BatchNorm2d(out_channels)
    self.act = Conv.default_act
    self.reset_parameters()

  def forward(self, x):
    offset_mask = self.conv_offset_mask(x)
    o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
    offset = torch.cat((o1, o2), dim=1)
    mask = torch.sigmoid(mask)
    x = torch.ops.torchvision.deform_conv2d(
      x,
      self.weight,
      offset,
      mask,
      self.bias,
      self.stride[0], self.stride[1],
      self.padding[0], self.padding[1],
      self.dilation[0], self.dilation[1],
      self.groups,
      self.deformable_groups,
      True
    )
    x = self.bn(x)
    x = self.act(x)
    return x

  def reset_parameters(self):
    n = self.in_channels
    for k in self.kernel_size:
      n *= k
    std = 1. / math.sqrt(n)
    self.weight.data.uniform_(-std, std)
    self.bias.data.zero_()
    self.conv_offset_mask.weight.data.zero_()
    self.conv_offset_mask.bias.data.zero_()



class PFLDInference(nn.Module):
    def __init__(self):
        super(PFLDInference, self).__init__()

        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.SiLU()

        self.conv2 = nn.Conv2d(64,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.SiLU()

        self.conv3_1 = InvertedResidual(64, 64, 2, False, 2)

        self.block3_2 = DCNv2(64, 64)
        self.block3_3 = DCNv2(64, 64)
        self.block3_4 = DCNv2(64, 64)
        self.block3_5 = DCNv2(64, 64)

        self.conv4_1 = InvertedResidual(64, 128, 2, False, 2)

        self.conv5_1 = DCNv2(128, 128)
        self.block5_2 = DCNv2(128, 128)
        self.block5_3 = DCNv2(128, 128)
        self.block5_4 = DCNv2(128, 128)
        self.block5_5 = DCNv2(128, 128)
        self.block5_6 = DCNv2(128, 128)

        self.conv6_1 = InvertedResidual(128, 16, 1, False, 2)  # [16, 14, 14]

        self.conv7 = Conv(16, 32, 3, 2, p=1)  # [32, 7, 7]
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)  # [128, 1, 1]
        self.bn8 = nn.BatchNorm2d(128)

        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(176, 196)

    def forward(self, x):  # x: 3, 112, 112
        x = self.relu(self.bn1(self.conv1(x)))  # [64, 56, 56]
        x = self.relu(self.bn2(self.conv2(x)))  # [64, 56, 56]
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)

        x = self.conv4_1(out1)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        x = self.conv6_1(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.relu(self.conv8(x))
        x3 = x3.view(x3.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)

        return out1, landmarks



class AuxiliaryNet(nn.Module):
    def __init__(self):
        super(AuxiliaryNet, self).__init__()
        self.conv1 = Conv(64, 128, 3, 2, p=1)
        self.conv2 = Conv(128, 128, 3, 1, p=1)
        self.conv3 = Conv(128, 32, 3, 2, p=1)
        self.conv4 = Conv(32, 128, 7, 1, p=1)
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


