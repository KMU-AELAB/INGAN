import torch
import numpy as np
import torch.nn as nn


class HourglassLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.h_loss = nn.L1Loss()

    def forward(self, recons, features, heights):
        cosine_distance = self.cosine_distance(features[0], features[1]) +\
                          (self.cosine_distance(features[0], features[2]) / 2) +\
                          (self.cosine_distance(features[0], features[3]) / 4)

        area_loss = self.area_loss(recons[0], recons[1]) + (self.area_loss(recons[0], recons[2]) / 2) +\
                    (self.area_loss(recons[0], recons[3]) / 4)

        h_loss = self.h_loss(heights[0].view(-1, 1), heights[1])
#         print(cosine_distance, area_loss, h_loss, '!!!!!!!!!!!')
        return cosine_distance * 2 + (area_loss * 0.0002) + (h_loss * 0.2)

    def area_loss(self, out, target):
        area = torch.mean(torch.abs(torch.sum(target, (1, 2, 3)) - torch.sum(out, (1, 2, 3))))
        
        return area

    def cosine_distance(self, feature1, feature2):
        x_len = torch.sqrt(torch.sum(feature1 * feature1, 1))
        y_len = torch.sqrt(torch.sum(feature2 * feature2, 1))
        inner_product = torch.sum(feature1 * feature2, 1)

        return torch.mean(1. - torch.div(inner_product, x_len * y_len + 1e-8))
