import torch
import numpy as np
import cv2
import torch.nn as nn


class HourglassLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, features, targets):
        area_loss = self.area_loss(outputs[0], targets[0]) + (self.area_loss(outputs[1], targets[0]) / 2) +\
                    (self.area_loss(outputs[2], targets[0]) / 4) + (self.area_loss(outputs[3], targets[0]) / 8)

        structural_loss = self.structural_loss(outputs[0], targets[0]) + (self.structural_loss(outputs[1], targets[0]) / 2) +\
                    (self.structural_loss(outputs[2], targets[0]) / 4) + (self.structural_loss(outputs[3], targets[0]) / 8)

        cosine_distance = self.cosine_distance(features[0], targets[1]) + (self.cosine_distance(features[1], targets[1]) / 2) +\
                    (self.cosine_distance(features[2], targets[1]) / 4) + (self.cosine_distance(features[3], targets[1]) / 8)

        return area_loss + structural_loss + cosine_distance

    def area_loss(self, out, target):
        return torch.sum(target) - torch.sum(out)

    def structural_loss(self, out, target):
        out = out.view(-1, 512, 512)
        target = target.view(-1, 512, 512)

        out_contours = cv2.findContours(out, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        target_contours = cv2.findContours(target, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

        return abs(out_contours - target_contours)

    def cosine_distance(self, feature1, feature2):
        x_len = torch.sqrt(torch.sum(feature1 * feature1, 1))
        y_len = torch.sqrt(torch.sum(feature2 * feature2, 1))
        inner_product = torch.sum(feature1 * feature2, 1)

        return 1. - torch.mean(torch.div(inner_product, x_len * y_len + 1e-8))