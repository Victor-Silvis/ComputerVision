
"""Script for Training the model"""

import warnings
import config
import torch
import torch.optim as optim
from model import Model
from loss import ModelLoss
from tqdm import tqdm
from utils import (
    mean_average_precision,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,)


warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0 = y[0].to(config.DEVICE)
        y1 = y[1].to(config.DEVICE)
        y2 = y[2].to(config.DEVICE)

        with torch.amp.autocast("cuda"):
            y_pred = model(x)
            loss = (
                loss_fn(y_pred[0], y0, scaled_anchors[0])
                + loss_fn(y_pred[1], y1, scaled_anchors[1])
                + loss_fn(y_pred[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)



def main():
    model = Model(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = ModelLoss()
    scaler = torch.amp.GradScaler("cuda")

    train_loader, test_loader = get_loaders(
        train_img_dir= config.TRAIN_IMG_DIR,
        train_lbl_dir= config.TRAIN_LABEL_DIR,
        test_img_dir= config.TEST_IMG_DIR,
        test_lbl_dir= config.TEST_LABEL_DIR,
        test_sample_size= config.TEST_SAMPLE_LIMIT)

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if config.SAVE_MODEL:
           save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

        if epoch > 0 and epoch % 10 == 0:
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            model.train()


if __name__ == "__main__":
    main()