import torch
import torch.nn as nn
import torch.nn.functional as F

from graph.weights_initializer import weights_init


class ResNetBlock(nn.Module):
    def __init__(self, _in, _out):
        super(ResNetBlock, self).__init__()

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


class ReduceBlock(nn.Module):
    def __init__(self, _in, _out):
        super(ReduceBlock, self).__init__()

        self.conv_seq = nn.Sequential(
            nn.Conv2d(_in, _in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(_in),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(_in, _in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(_in),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(_in, _out, kernel_size=3, stride=[2, 1], padding=1, bias=False),
            nn.BatchNorm2d(_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.apply(weights_init)

    def forward(self, x):
        return self.conv_seq(x)


class Reduce(nn.Module):
    def __init__(self, _in):
        super(Reduce, self).__init__()

        self.conv_seq = nn.Sequential(
            ReduceBlock(_in, _in // 2),
            ReduceBlock(_in // 2, _in // 2),
            ReduceBlock(_in // 2, _in // 4),
            ReduceBlock(_in // 4, _in // 8),
        )

        self.apply(weights_init)

    def forward(self, x):
        return self.conv_seq(x)


class FeatureExtract(nn.Module):
    def __init__(self):
        super(FeatureExtract, self).__init__()

        self.block0 = ResNetBlock(3, 128)
        self.block1 = ResNetBlock(128, 256)
        self.block2 = ResNetBlock(256, 512)
        self.block3 = ResNetBlock(512, 1024)
        self.block4 = ResNetBlock(1024, 2048)

        self.reduce1 = Reduce(256)
        self.reduce2 = Reduce(512)
        self.reduce3 = Reduce(1024)
        self.reduce4 = Reduce(2048)

        self.apply(weights_init)

    def forward(self, x):
        inter = self.block0(x)

        inter_128 = self.block1(inter)
        inter_64 = self.block2(inter_128)
        inter_32 = self.block3(inter_64)
        inter_16 = self.block4(inter_32)
        
        reduce1 = self.reduce1(inter_128).view(-1, 256, 1, 256)
        reduce2 = F.interpolate(self.reduce2(inter_64), size=(4, 256),
                                mode='bilinear', align_corners=False).view(-1, 256, 1, 256)
        reduce3 = F.interpolate(self.reduce3(inter_32), size=(2, 256),
                                mode='bilinear', align_corners=False).view(-1, 256, 1, 256)
        reduce4 = F.interpolate(self.reduce4(inter_16), size=(1, 256),
                                mode='bilinear', align_corners=False).view(-1, 256, 1, 256)
        
        feature = torch.cat((reduce1, reduce2, reduce3, reduce4), dim=1)    # 1024 1 256
        feature = feature.reshape(-1, 1024, 256).permute(2, 0, 1)

        return feature


class HorizonBase(nn.Module):
    def __init__(self):
        super(HorizonBase, self).__init__()

        self.feature_extract = FeatureExtract()

        self.lstm = nn.LSTM(input_size=1024,
                            hidden_size=512,
                            num_layers=2,
                            dropout=0.5,
                            batch_first=False,
                            bidirectional=True)
        self.drop_out = nn.Dropout(0.5)

        self.linear = nn.Linear(in_features=2 * 512,
                                out_features=12)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.apply(weights_init)

    def forward(self, x):
        batch_size = x.size(0)

        feature = self.feature_extract(x)
        
        output, _ = self.lstm(feature)  # 256 b 1024
        output = self.drop_out(output)

        output = self.lrelu(self.linear(output))
        output = output.view(256, batch_size, 3, 4) # 256 b 12
        output = output.permute(1, 2, 0, 3) # b 3 256 4

        return output   # b 3 256 4


class Corner(nn.Module):
    def __init__(self):
        super(Corner, self).__init__()

        self.conv1 = nn.Conv2d(3, 2, kernel_size=[1, 3], stride=1, padding=[0, 1], bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=[1, 3], stride=1, padding=[0, 1], bias=False)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, horizon_output):
        horizon_output = horizon_output.contiguous().view(horizon_output.size(0), 3, 1, -1) # b 3 1 1024
        
        corner = self.lrelu(self.conv1(horizon_output))
        corner = self.sigmoid(self.conv2(corner))
        
        return corner.view(-1, 1, 1024)


class FloorMap(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv_1 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=[1, 8], stride=[1, 8], bias=False)
        self.deconv_2 = nn.ConvTranspose2d(in_channels=3, out_channels=2, kernel_size=[1, 8], stride=[1, 8], bias=False)
        self.deconv_3 = nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=4, stride=2,
                                           padding=1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.apply(weights_init)

    def forward(self, horizon_output):
        output = self.lrelu(self.deconv_1(horizon_output))
        output = self.lrelu(self.deconv_2(output))
        output = self.sigmoid(self.deconv_3(output))

        return output