import torch
import numpy as np
import torch.nn as nn


class HourglassLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.BCELoss()
        self.h_loss = nn.L1Loss()

    def forward(self, recons, heights):
        area_loss = self.loss(recons[1], recons[0]) + (self.loss(recons[2], recons[0]) / 2) +\
                    (self.loss(recons[3], recons[0]) / 4)

        h_loss = self.h_loss(heights[1], heights[0].view(-1, 1))
        
        return (area_loss) + (h_loss)*0.01