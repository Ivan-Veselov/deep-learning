import torch
from torch import nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, cardinality, downsampling_stride):
        super(Bottleneck, self).__init__()

        self.__shortcut_projection = None

        out_channels_halved = out_channels // 2
        middle_convolution_stride = 1
        if downsampling_stride is not None:
            middle_convolution_stride = downsampling_stride
            self.__shortcut_projection = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=downsampling_stride,
                bias=False
            )
        else:
            if in_channels != out_channels:
                self.__shortcut_projection = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                )

        middle_convolution = nn.Conv2d(
            out_channels_halved,
            out_channels_halved,
            kernel_size=3,
            padding=1,
            stride=middle_convolution_stride,
            groups=cardinality,
            bias=False
        )

        self.__relu = nn.ReLU()
        self.__residual_mapping = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_halved, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels_halved),
            self.__relu,
            middle_convolution,
            nn.BatchNorm2d(out_channels_halved),
            self.__relu,
            nn.Conv2d(out_channels_halved, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        shortcut = x
        if self.__shortcut_projection is not None:
            shortcut = self.__shortcut_projection(shortcut)

        out = self.__residual_mapping(x) + shortcut
        out = self.__relu(out)

        return out


class ResNeXt(torch.nn.Module):
    in_channels = 3
    in_map_size = 224
    classes_num = 10

    def __init__(self, layers_size, cardinality):
        assert len(layers_size) == 4
        super(ResNeXt, self).__init__()

        self.__cardinality = cardinality

        channels = 64
        map_size = self.in_map_size

        self.__relu = nn.ReLU()
        modules = [
            nn.Conv2d(self.in_channels, channels, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(channels),
            self.__relu,
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        ]

        map_size //= 4
        modules += self.__make_block(channels, 4 * channels, layers_size[0], None)
        channels *= 4

        for blocks_num in layers_size[1:]:
            modules += self.__make_block(channels, 2 * channels, blocks_num, 2)
            channels *= 2
            map_size //= 2

        modules.append(nn.AvgPool2d(map_size, stride=1))
        self.__convolution = nn.Sequential(*modules)

        self.__linear = nn.Linear(channels, self.classes_num)

    def forward(self, x):
        out = self.__convolution(x)
        out = out.view(out.size(0), -1)
        out = self.__linear(out)

        if self.training:
            return out

        return self.softmax(out)

    def __make_block(self, in_channels, out_channels, num_of_blocks, downsampling_stride):
        return [Bottleneck(in_channels, out_channels, self.__cardinality, downsampling_stride)] + \
            [Bottleneck(out_channels, out_channels, self.__cardinality, None) for _ in range(num_of_blocks - 1)]
