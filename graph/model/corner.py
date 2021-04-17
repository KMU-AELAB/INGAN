import torch
import torch.nn as nn
import torch.nn.functional as F

from graph.weights_initializer import weights_init


class Down(nn.Module):
    def __init__(self, _in, _out):
        super(Down, self).__init__()

        self.conv_seq = nn.Sequential(
            nn.Conv2d(_in, _in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(_in),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(_in, _in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(_in),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(_in, _out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.apply(weights_init)

    def forward(self, x):
        return self.conv_seq(x)


class CornerModule(nn.Module):
    def __init__(self):
        super(CornerModule, self).__init__()

        self.down1 = Down(3, 32)  # 512 -> 256
        self.down2 = Down(32, 64)  # 256 -> 128
        self.down3 = Down(64, 128)  # 128 -> 64
        self.down4 = Down(128, 192)  # 64 -> 32

        self.global_conv = nn.Conv2d(192, 64, kernel_size=[32, 1], stride=1, bias=False)

        self.up = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=[1, 4], stride=2,
                                     padding=1, bias=False)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        
        z = self.lrelu(self.global_conv(down4))
        
        up = self.sigmoid(self.up(z))
        
        return up


class Corner(nn.Module):
    def __init__(self):
        super().__init__()

        self.corner = CornerModule()

        self.apply(weights_init)

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim=1)
        out = self.corner(x)

        return out.view(-1, 1, 1024)