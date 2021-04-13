import torch
import torch.nn as nn
import torch.nn.functional as F

from graph.weights_initializer import weights_init


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_conv1 = nn.Conv2d(512, 640, kernel_size=[16, 32], stride=1, bias=False)
        self.feature_conv2 = nn.Conv2d(512, 640, kernel_size=16, stride=1, bias=False)
        self.feature_conv3 = nn.Conv2d(512, 640, kernel_size=16, stride=1, bias=False)

        self.linear1 = nn.Linear(640 * 3, 512)
        self.linear2 = nn.Linear(512, 1, bias=True)

        self.dropout = nn.Dropout(p=0.4)

        self.relu = nn.ReLU(inplace=True)

        self.apply(weights_init)

    def forward(self, x1, x2, x3):
        feature1 = self.feature_conv1(x1)
        feature2 = self.feature_conv1(x2)
        feature3 = self.feature_conv1(x3)

        feature = torch.cat((feature1, feature2, feature3), dim=1).view(-1, 640 * 3)

        x = self.relu(self.linear1(feature))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        out = self.dropout(x)
        
        return out