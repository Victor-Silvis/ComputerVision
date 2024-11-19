'''Script to calculate loss'''

import torch
import torch.nn as nn
from utils import intersection_over_union

class ModelLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

        #Loss Weights
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true, anchors):
        '''
        Note:
        anchors = Scaled Anchors (Relative to Cell instead of image)
        Box coords = Use offset denormalization as specified by paper'''

        #Create Masks (Object and No Object)
        obj = y_true[..., 0] == 1
        noobj = y_true[..., 0] == 0

        #No object Loss
        no_object_loss = self.bce((y_pred[..., 0:1][noobj]), (y_true[..., 0:1][noobj]),)

        #Object Loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(y_pred[..., 1:3]), torch.exp(y_pred[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], y_true[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(y_pred[..., 0:1][obj]), ious * y_true[..., 0:1][obj])

        #Box Coords Loss
        y_pred[..., 1:3] = self.sigmoid(y_pred[..., 1:3])
        y_true[..., 3:5] = torch.log((1e-16 + y_true[..., 3:5] / anchors))
        box_loss = self.mse(y_pred[..., 1:5][obj], y_true[..., 1:5][obj])

        #Class Loss
        class_loss = self.entropy((y_pred[..., 5:][obj]), (y_true[..., 5][obj].long()),)

        #Return Loss * Weights
        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss)