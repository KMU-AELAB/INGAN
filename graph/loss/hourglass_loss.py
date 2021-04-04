import torch
import numpy as np
import torch.nn as nn


class HourglassLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, outputs, features, targets):
        # structural_loss = self.structural_loss(outputs[0], targets[0]) + (self.structural_loss(outputs[1], targets[0]) / 2) +\
        #             (self.structural_loss(outputs[2], targets[0]) / 4) + (self.structural_loss(outputs[3], targets[0]) / 8)
        area_loss = self.area_loss(outputs[0], targets[0]) + (self.area_loss(outputs[1], targets[0]) / 2) +\
                    (self.area_loss(outputs[2], targets[0]) / 4)

        cosine_distance = self.cosine_distance(features[0], targets[1]) + (self.cosine_distance(features[1], targets[1]) / 2) +\
                    (self.cosine_distance(features[2], targets[1]) / 4)
        print(area_loss, cosine_distance, '?????????????')
        return cosine_distance + area_loss * 0.001 #+ structural_loss

    def area_loss(self, out, target):
        area = torch.mean(torch.abs(torch.sum(target, (1, 2, 3)) - torch.sum(out, (1, 2, 3))))
        sub_area = abs(torch.nonzero(target).size(0) - torch.nonzero(out).size(0)) / target.size(0)
        
        return area + (sub_area * 0.002)

    def cosine_distance(self, feature1, feature2):
        x_len = torch.sqrt(torch.sum(feature1 * feature1, 1))
        y_len = torch.sqrt(torch.sum(feature2 * feature2, 1))
        inner_product = torch.sum(feature1 * feature2, 1)

        return (1. - torch.mean(torch.div(inner_product, x_len * y_len + 1e-8))) * 1000

    # def structural_loss(self, out, target):
    #     out = out.view(-1, 512, 512)
    #     target = target.view(-1, 512, 512)
    #
    #     out_contours = cv2.findContours(out, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    #     target_contours = cv2.findContours(target, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    #
    #     return abs(out_contours - target_contours)