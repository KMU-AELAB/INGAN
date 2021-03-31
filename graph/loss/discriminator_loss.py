import torch
import numpy as np
import torch.nn as nn


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.BCELoss()
        self.tm_loss = nn.TripletMarginLoss()

    def forward(self, origin, var1, var2, f):
        distance_loss = self.cosine_distance(origin, var1) + self.cosine_distance(origin, var2)
        triple_loss = self.tm_loss(origin, var1, f) + self.tm_loss(origin, var2, f)
        
        return distance_loss + triple_loss*1.5
    
    def cosine_distance(self, feature1, feature2):
        x_len = torch.sqrt(torch.sum(feature1 * feature1, 1))
        y_len = torch.sqrt(torch.sum(feature2 * feature2, 1))
        inner_product = torch.sum(feature1 * feature2, 1)
        
        return 1. - torch.mean(torch.div(inner_product, x_len * y_len + 1e-8))