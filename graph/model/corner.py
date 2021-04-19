import torch
import torch.nn as nn
import torch.nn.functional as F

from graph.weights_initializer import weights_init


class Up(nn.Module):
    def __init__(self, _in, _out):
        super(Up, self).__init__()

        self.deconv = nn.ConvTranspose2d(in_channels=_in, out_channels=_out, kernel_size=[1, 4], stride=[1, 2],
                                         padding=[0, 1], bias=False)
        self.norm = nn.BatchNorm2d(_out)

        self.apply(weights_init)

    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)

        return x


class CornerModule(nn.Module):
    def __init__(self):
        super(CornerModule, self).__init__()

        self.deconv = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=[1, 32], stride=[1, 32],
                                         bias=False)
        
        self.up1 = Up(1536, 592)  # 16 -> 32
        self.up2 = Up(592, 192)  # 32 -> 64
        self.up3 = Up(192, 64)  # 64 -> 128
        self.up4 = Up(64, 32)  # 128 -> 256
        self.up5 = Up(32, 1)  # 256 -> 512

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x1, x2, x3):
        x1 = self.lrelu(self.deconv(x1))
        x2 = self.lrelu(self.deconv(x2))
        x3 = self.lrelu(self.deconv(x3))
        
        x = torch.cat((x1, x2, x3), dim=1)
        
        up1 = self.lrelu(self.up1(x))
        up2 = self.lrelu(self.up2(up1))
        up3 = self.lrelu(self.up3(up2))
        up4 = self.lrelu(self.up4(up3))
        up5 = self.sigmoid(self.up5(up4))
        
        return up5


class Corner(nn.Module):
    def __init__(self):
        super().__init__()

        self.corner = CornerModule()

        self.apply(weights_init)

    def forward(self, x1, x2, x3):
        out = self.corner(x1, x2, x3)

        return out.view(-1, 1, 1024)