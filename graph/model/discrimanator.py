import torch
import torch.nn as nn
import torch.nn.functional as F

from graph.weights_initializer import weights_init


def conv2d(_in):
    return nn.Sequential(
        nn.Conv2d(_in, _in, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(_in),
        nn.ReLU(inplace=True),
        nn.Conv2d(_in, _in, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(_in),
        nn.ReLU(inplace=True),
    )


def reduce2d(_in, _out):
    return nn.Sequential(
        nn.Conv2d(_in, _out, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(_out),
        nn.ReLU(inplace=True),
    )

class DiscriminatorModule(nn.Module):
    def __init__(self, channel_size):
        super(DiscriminatorModule, self).__init__()

        self.channel_size = channel_size
        self.conv_lst = nn.ModuleList([conv2d(i) for i in self.channel_size[:-1]])

        self.reduce_lst = nn.ModuleList(
            [reduce2d(self.channel_size[i - 1], self.channel_size[i]) for i in range(1, len(self.channel_size))]
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_linear = nn.Linear(self.channel_size[-1], 192)

        self.lrelu = nn.LeakyReLU(inplace=True)

        self.apply(weights_init)

    def forward(self, x):
        for conv, reduce in zip(self.conv_lst, self.reduce_lst):
            _x = conv(x)
            x = x + _x
            x = reduce(x)

        x = self.avg_pool(x)

        feature = self.lrelu(self.feature_linear(x)).view(-1, 192)

        return feature


class Discriminator(nn.Module):
    def __init__(self, channel_size=[1, 32, 64, 128, 256, 512, 1024, 2048]):
        super(Discriminator, self).__init__()

        self.channel_size = channel_size

        self.discriminator = DiscriminatorModule(channel_size)

        self.apply(weights_init)

    def forward(self, x):
        feature = self.discriminator(x)

        return feature


# class Discriminator(nn.Module):
#     def __init__(self, channel_size=[1, 32, 64, 128, 256, 512, 1024, 2048]):
#         super(Discriminator, self).__init__()
#
#         self.channel_size = channel_size
#
#         self.discriminator = DiscriminatorModule(channel_size)
#         self.out_linear = nn.Linear(192*2, 1)
#
#         self.sigmoid = nn.Sigmoid()
#
#         self.apply(weights_init)
#
#     def forward(self, x1, x2):
#         feature_x1 = self.discriminator(x1)
#         feature_x2 = self.discriminator(x2)
#
#         x = torch.cat((feature_x1, feature_x2), dim=1)
#
#         output = self.sigmoid(self.out_linear(x))
#
#         return output, feature_x1, feature_x2