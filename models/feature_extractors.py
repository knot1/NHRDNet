import torch
import torch.nn as nn

class Conv2dRelu(nn.Module):
    """Block holding one Conv2d and one ReLU layer"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._conv = nn.Conv2d(*args, **kwargs)
        self._relu = nn.ReLU()

    def forward(self, input_batch):
        return self._relu(self._conv(input_batch))


class Img224x224Kernel7x7SeparatedDims(nn.Module):
    def __init__(self, in_channels, out_params, norm_class=None, p_dropout=0.0):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels (e.g., 3 for RGB images)
        out_params : int
            Number of output parameters
        norm_class : nn.Module, optional
            Normalization layer (e.g., nn.BatchNorm2d), by default None
        p_dropout : float, optional
            Dropout probability, by default 0.0
        """
        super().__init__()

        layers = []
        layers.append(Conv2dRelu(in_channels, 64, (7, 1)))
        layers.append(Conv2dRelu(64, 64, (1, 7)))

        layers.append(Conv2dRelu(64, 128, (7, 7), stride=2))
        if norm_class:
            layers.append(norm_class(128))
        if p_dropout > 0:
            layers.append(nn.Dropout2d(p_dropout))

        layers.append(Conv2dRelu(128, 128, (7, 1)))
        layers.append(Conv2dRelu(128, 128, (1, 7)))

        layers.append(Conv2dRelu(128, 256, (7, 7), stride=2))
        if norm_class:
            layers.append(norm_class(256))
        if p_dropout > 0:
            layers.append(nn.Dropout2d(p_dropout))

        layers.append(Conv2dRelu(256, 256, (5, 1)))
        layers.append(Conv2dRelu(256, 256, (1, 5)))

        layers.append(Conv2dRelu(256, 256, (5, 5), stride=2))
        if norm_class:
            layers.append(norm_class(256))
        if p_dropout > 0:
            layers.append(nn.Dropout2d(p_dropout))

        layers.append(Conv2dRelu(256, 256, (5, 1)))
        layers.append(Conv2dRelu(256, 256, (1, 5)))

        layers.append(Conv2dRelu(256, 128, (5, 5), stride=2))
        if norm_class:
            layers.append(norm_class(128))
        if p_dropout > 0:
            layers.append(nn.Dropout2d(p_dropout))

        layers.append(Conv2dRelu(128, 128, (3, 1)))
        layers.append(Conv2dRelu(128, 128, (1, 3)))
        layers.append(Conv2dRelu(128, 128, (3, 1)))
        layers.append(Conv2dRelu(128, 128, (1, 3)))

        layers.append(nn.Conv2d(128, out_params, (2, 2)))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
