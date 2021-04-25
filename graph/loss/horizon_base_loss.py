import torch
import numpy as np
import torch.nn as nn


class HorizonBaseLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.BCELoss()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, target, output):
        loss = self.loss(output, target)

        ones, zeros = torch.ones(target.size()).cuda(), torch.zeros(target.size()).cuda()
        sub_loss = self.loss(torch.where(output > 0.5, ones, zeros),
                             target)

        return loss + sub_loss * 0.2
