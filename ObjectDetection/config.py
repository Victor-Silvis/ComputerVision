'''Main Config File'''

#packages
import os
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

#Config Settings
DATASET = 'Dataset'
NUM_WORKERS = 4
BATCH_SIZE = 32
IMAGE_SIZE = 416
NUM_CLASSES =  12
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.65
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_FILE = "checkpoint.pth.tar"
TRAIN_DIR = 'train'
TEST_DIR = 'test'
LABELS_DIR = 'labels'
IMAGES_DIR = 'images'
TEST_SAMPLE_LIMIT = 30
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Pre-set Anchors
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

#Directories
TRAIN_IMG_DIR = os.path.join(DATASET, TRAIN_DIR, IMAGES_DIR)
TRAIN_LABEL_DIR = os.path.join(DATASET, TRAIN_DIR, LABELS_DIR)
TEST_IMG_DIR = os.path.join(DATASET, TEST_DIR, IMAGES_DIR)
TEST_LABEL_DIR = os.path.join(DATASET, TEST_DIR, LABELS_DIR)

#Dataset Classes (Example)
DATASET_CLASSES = [
    'Aircraft',
    'Stairs',
    'Ground Staff',
    'Pylon',
    'GPU',
    'Catering',
    'Bus',
    'PAX',
    'Tug',
    'Service Car',
    'Pushback',
    'Bagcart',
    'Belt Loader',
    'Fuel Truck',
    'Refuel Hose'
]


#Image Preproccesing logic
scale = 1.1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)
                ),
                A.Affine(shear=15, p=0.5, mode=cv2.BORDER_CONSTANT),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)


test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, 
            min_width=IMAGE_SIZE, 
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)