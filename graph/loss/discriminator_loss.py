import torch
import numpy as np
import torch.nn as nn


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.BCELoss()
        self.tm_loss = nn.TripletMarginLoss()

    def forward(self, features):
        tmp = self.cosine_distance(features[0], features[1]) + self.cosine_distance(features[0], features[2])
        molecule = tmp + self.cosine_distance(features[1], features[2]) * 0.01
        denominator = self.cosine_distance(features[3], features[1]) + self.cosine_distance(features[3], features[2]) + 1e-5
        
        return (molecule / denominator) + (tmp * 0.1), tmp, denominator # triple_loss + (distance_loss * 10) + (gan_loss * 0.1)
    
    def cosine_distance(self, feature1, feature2):
        x_len = torch.sqrt(torch.sum(feature1 * feature1, 1))
        y_len = torch.sqrt(torch.sum(feature2 * feature2, 1))
        inner_product = torch.sum(feature1 * feature2, 1)
        
        return torch.mean(1. - torch.div(inner_product, x_len * y_len + 1e-8))