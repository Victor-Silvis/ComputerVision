
'''Script to load raw data and used to feed the model'''

#Imports
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from utils import (iou_width_height as iou)
ImageFile.LOAD_TRUNCATED_IMAGES = True


#Dataset Class
class ModelDataloader(Dataset):

    def __init__(self, img_dir, label_dir, anchors, 
                 image_size, S, C, transform = None):
        
        '''
        args:
        img_dir     :   Location of the images
        label_dir   :   Location of the labels
        anchors     :   Defined Anchors (Relative to image)
        image_size  :   Image size
        S           :   Defined Scales [13, 26, 52] by default
        C           :   Number of classes in dataset
        transform   :   Tranform functions (e.g. train or test)
        '''

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.C = C
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5
        self.images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        #Gets Image/Label from raw data and preprocesses
        img_filename = self.images[index]
        img_path = os.path.join(self.img_dir, img_filename)
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_filename)
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        
        #Init target tensors for each scale, and rest of code below fills the targets
        targets = [torch.zeros((self.num_anchors_per_scale, S, S, 6)) for S in self.S]

        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3
            
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = width * S, height * S
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)
