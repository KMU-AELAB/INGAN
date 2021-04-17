import torch
import torch.nn as nn
import torch.nn.functional as F

from graph.weights_initializer import weights_init


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear0_1 = nn.Linear(1024, 384)
        self.linear0_2 = nn.Linear(1024, 384)
        self.linear0_3 = nn.Linear(1024, 384)

        self.linear1 = nn.Linear(384 * 3, 384)
        self.linear2 = nn.Linear(384, 1, bias=True)

        self.dropout = nn.Dropout(p=0.4)

        self.relu = nn.ReLU(inplace=True)

        self.apply(weights_init)

    def forward(self, x1, x2, x3):
        feature1 = self.relu(self.linear0_1(x1.view(-1, 1024)))
        feature1 = self.dropout(feature1)
        
        feature2 = self.relu(self.linear0_1(x2.view(-1, 1024)))
        feature2 = self.dropout(feature2)
        
        feature3 = self.relu(self.linear0_1(x3.view(-1, 1024)))
        feature3 = self.dropout(feature3)
        
        feature = torch.cat((feature1, feature2, feature3), dim=1)

        x = self.relu(self.linear1(feature))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        out = self.dropout(x)
        
        return out