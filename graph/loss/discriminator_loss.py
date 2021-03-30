import torch
import numpy as np
import torch.nn as nn


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.BCELoss()

    def forward(self, feature1, feature2):
        x_len = torch.sqrt(torch.sum(feature1 * feature1, 1))
        y_len = torch.sqrt(torch.sum(feature2 * feature2, 1))
        inner_product = torch.sum(feature1 * feature2, 1)
        result = torch.div(inner_product, x_len * y_len + 1e-8)

        return 1. - result
