import torch
import numpy as np
import torch.nn as nn


class HorizonBaseLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.BCELoss()

    def forward(self, target, output):
        return self.loss(output, target)
