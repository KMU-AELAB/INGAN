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
            nn.ReLU(inplace=True),
            nn.Conv2d(_in, _in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(_in, _out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(_out),
            nn.ReLU(inplace=True),
        )

        self.apply(weights_init)

    def forward(self, x):
        return self.conv_seq(x)

class Up(nn.Module):
    def __init__(self, _in, _out):
        super(Up, self).__init__()

        self.deconv = nn.ConvTranspose2d(in_channels=_in, out_channels=_out, kernel_size=4, stride=2,
                                         padding=1, bias=False)
        self.conv = nn.Conv2d(_in * 2, _in, kernel_size=3, stride=1, padding=1, bias=False)

        self.lrelu = nn.LeakyReLU(inplace=True)

        self.apply(weights_init)

    def forward(self, x1, x2):
        # add padding
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)

        x = self.lrelu(self.conv(x))
        x = self.deconv(x)

        return x

class Hourglass(nn.Module):
    def __init__(self):
        super(Hourglass, self).__init__()

        self.down1 = Down(1, 32)    # 512 -> 256
        self.down2 = Down(32, 64)    # 256 -> 128
        self.down3 = Down(64, 128)    # 128 -> 64
        self.down4 = Down(128, 256)    # 64 -> 32
        self.down5 = Down(256, 512)    # 32 -> 16

        self.global_conv = nn.Conv2d(512, 1024, kernel_size=16, stride=1, bias=False)
        self.deconv = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=[16,16], stride=[16, 16],
                                         bias=False)

        self.up5 = Up(512, 256)    # 16 -> 32
        self.up4 = Up(256, 128)    # 32 -> 64
        self.up3 = Up(128, 64)    # 64 -> 128
        self.up2 = Up(64, 32)    # 128 -> 256
        self.up1 = Up(32, 1)    # 256 -> 512

        self.lrelu = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)

        self.apply(weights_init)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)

        z = self.lrelu(self.global_conv(down5))
        up_in = self.deconv(z)

        up5 = self.lrelu(self.up5(up_in, down5))
        up4 = self.lrelu(self.up4(up5, down4))
        up3 = self.lrelu(self.up3(up4, down3))
        up2 = self.lrelu(self.up2(up3, down2))
        up1 = self.up1(up2, down1)
        
        return up1


class InHourglass(nn.Module):
    def __init__(self):
        super(InHourglass, self).__init__()

        self.down1 = Down(3, 32)  # 512 -> 256
        self.down2 = Down(32, 64)  # 256 -> 128
        self.down3 = Down(64, 128)  # 128 -> 64
        self.down4 = Down(128, 256)  # 64 -> 32
        self.down5 = Down(256, 512)  # 32 -> 16

        self.global_conv = nn.Conv2d(512, 1024, kernel_size=[32, 16], stride=1, bias=False)
        self.deconv = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=[16, 16], stride=[16, 16],
                                         bias=False)
        
        self.up5 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2,
                                      padding=1, bias=False)
        self.up4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2,
                                      padding=1, bias=False)
        self.up3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2,
                                      padding=1, bias=False)
        self.up2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2,
                                      padding=1, bias=False)
        self.up1 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2,
                                      padding=1, bias=False)

        self.lrelu = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)

        self.apply(weights_init)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)

        z = self.lrelu(self.global_conv(down5))
        up_in = self.deconv(z)

        up5 = self.relu(self.up5(up_in))
        up4 = self.relu(self.up4(up5))
        up3 = self.relu(self.up3(up4))
        up2 = self.relu(self.up2(up3))
        up1 = self.up1(up2)
        
        return up1


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_hourglass = InHourglass()
        self.hourglass1 = Hourglass()
        self.hourglass2 = Hourglass()

        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x):
        out1 = self.sigmoid(self.in_hourglass(x))
        out2 = self.sigmoid(self.hourglass1(out1))
        output = self.sigmoid(self.hourglass2(out2))
        
        return (output, out2, out1)