import torch
import numpy as np
import torch.nn as nn


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.BCELoss()
        self.tm_loss = nn.TripletMarginLoss()

    def forward(self, features, outs):
        valid = torch.Variable(torch.Tensor(outs[0].size(0), 1).fill_(1.0), requires_grad=False)
        fake = torch.Variable(torch.Tensor(outs[0].size(0), 1).fill_(0.0), requires_grad=False)

        distance_loss = self.cosine_distance(features[0], features[1]) + self.cosine_distance(features[0], features[2])
        triple_loss = self.tm_loss(features[0], features[1], features[3]) + self.tm_loss(features[0], features[2], features[3])
        gan_loss = self.loss(outs[0], valid) + self.loss(outs[1], valid) + \
                   self.loss(outs[2], valid) + (self.loss(outs[3], fake) * 3)
        print(distance_loss, triple_loss, gan_loss)
        return distance_loss + triple_loss*1.5
    
    def cosine_distance(self, feature1, feature2):
        x_len = torch.sqrt(torch.sum(feature1 * feature1, 1))
        y_len = torch.sqrt(torch.sum(feature2 * feature2, 1))
        inner_product = torch.sum(feature1 * feature2, 1)
        
        return 1. - torch.mean(torch.div(inner_product, x_len * y_len + 1e-8))