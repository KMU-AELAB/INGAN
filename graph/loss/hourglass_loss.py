import torch
import numpy as np
import torch.nn as nn


class HourglassLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.BCELoss()
        self.sub_loss = nn.BCEWithLogitsLoss()

    def forward(self, output, logit1, logit2, logit3, target):
        return self.loss(output, target) + (self.sub(logit1, target) / 2) + (self.sub(logit2, target) / 4) +\
               (self.sub(logit3, target) / 8)
