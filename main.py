# Basic and ML imports
import os
import numpy as np
import pandas as pd
from skimage import io
from skimage.transform import resize
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from torchvision import models, transforms
from sklearn.metrics import average_precision_score
import json
from torchmetrics.detection import MeanAveragePrecision


# For Mask R-CNN
import torchvision
import torchvision.models.detection.mask_rcnn
import torch.optim as optim
from tqdm import tqdm

# Metrics and utilities
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score  # For IoU


# Ensure reproducibility
torch.manual_seed(123)
np.random.seed(123)


class CustomDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        """
        Args:
            dataframe (DataFrame): DataFrame containing image info and annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 0])
        image = io.imread(img_name)
        # image = np.dstack([image, image, image])  # Convert to 3 channels if needed
        if len(image.shape) == 2:
            image = np.dstack([image, image, image])  # Convert grayscale to 3 channels
        elif len(image.shape) == 3 and image.shape[2] == 1:
        # If image has 3 dimensions but the third dimension is 1, it's also grayscale
            image = np.squeeze(image)  # Remove the third dimension
            image = np.dstack([image, image, image])
    
        bbox_str = self.dataframe.iloc[idx, 4]
        bbox = eval(bbox_str) if bbox_str != 'none' else None

        if bbox is not None:
            # Convert dict bbox to [xmin, ymin, xmax, ymax]
            bbox = [bbox['xmin'], bbox['ymin'], bbox['xmin'] + bbox['width'], bbox['ymin'] + bbox['height']]
            # Mock a binary mask for example purposes
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        target = {}
        target["boxes"] = torch.as_tensor([bbox], dtype=torch.float32) if bbox is not None else torch.as_tensor([[0,0,1,1]], dtype=torch.float32)
        # target["boxes"] = torch.as_tensor([bbox], dtype=torch.float32) if bbox is not None else torch.zeros((0, 4), dtype=torch.float32)
        target["labels"] = torch.as_tensor([1], dtype=torch.int64) if bbox is not None else torch.as_tensor([0], dtype=torch.int64)
        target["masks"] = torch.as_tensor(mask, dtype=torch.uint8).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image, target

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Load the dataset
df = pd.read_csv('../../scratch/hmudigon/datasets/TBX11K/data.csv')
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['target'])

# Assuming images are stored in 'train/' for both train and validation
train_dataset = CustomDataset(dataframe=train_df, img_dir='../../scratch/hmudigon/datasets/TBX11K/train/', transform=transform)
val_dataset = CustomDataset(dataframe=val_df, img_dir='../../scratch/hmudigon/datasets/TBX11K/train/', transform=transform)

def collate_fn(batch):
    """
    Custom collate function to handle batches where some images do not have associated bounding boxes.
    """
    images = [item[0] for item in batch]
    targets = []
    for item in batch:
        target = {}
        target["boxes"] = torch.as_tensor(item[1]["boxes"], dtype=torch.float32)
        target["labels"] = torch.as_tensor(item[1]["labels"], dtype=torch.int64)
        target["masks"] = torch.as_tensor(item[1]["masks"], dtype=torch.uint8)
        targets.append(target)
    
    images = torch.stack(images, 0)
    return images, targets

def bbox_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters
    ----------
    box1 : array-like of float
        (x_min, y_min, x_max, y_max) for the first bounding box.
    box2 : array-like of float
        (x_min, y_min, x_max, y_max) for the second bounding box.
    
    Returns
    -------
    float
        in [0, 1]. Intersection over union of the two bounding boxes.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute the union area by using both areas minus the intersection area
    union_area = box1_area + box2_area - intersection_area
    
    # Compute the IoU
    iou = intersection_area / union_area
    
    return iou

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=16, collate_fn=collate_fn)

# Assuming you are using the train_loader from the previous setup
for images, targets in train_loader:
    # images is a tensor of shape [B, C, H, W]
    batch_size = images.shape[0]
    channels = images.shape[1]
    height = images.shape[2]
    width = images.shape[3]

    print(f"Batch size: {batch_size}")
    print(f"Channels: {channels}")
    print(f"Image width: {width}")
    print(f"Image height: {height}")
    break  # Remove this break to print for all batches, keep it to print just the first batch

def get_model_instance_segmentation(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one (note num_classes includes the background)
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    # Now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256  # This is a typical number of features used
    
    # And replace the mask predictor with a new one
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model

# Adjust num_classes to match your dataset (Background + number of classes)
num_classes = 2
model = get_model_instance_segmentation(num_classes)

# Move model to the right device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Parameters
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# Learning rate scheduler which decreases the learning rate by 10x every 3 epochs
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

def validate_model(model, data_loader, device):
    model.eval()  # Set model to evaluation mode
    ious = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validating", leave=True):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            output = model(images)

            for i, out in enumerate(output):
                pred_boxes = out['boxes'].cpu().numpy()
                true_boxes = targets[i]['boxes'].cpu().numpy()

                # Calculate IoU for each predicted box with its corresponding ground truth box
                for pred_box, true_box in zip(pred_boxes, true_boxes):
                    iou = bbox_iou(pred_box, true_box)
                    ious.append(iou)

    # Calculate average IoU over all validation images
    avg_iou = sum(ious) / len(ious) if ious else 0
    return avg_iou

def validate_model_with_map(model, data_loader, device, iou_threshold=0.5):
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[iou_threshold])
    
    model.eval()  # Set model to evaluation mode
    all_detections = []  # Store all detections across the dataset
    all_annotations = []  # Store all ground truths across the dataset

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Calculating mAP", leave=True):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                true_boxes = targets[i]['boxes'].cpu().numpy()
                
                all_detections.append(output)
                all_annotations.append(targets[i])

    metric.update(all_detections, all_annotations)
    mAP = metric.compute()
    
    return mAP["map_50"].item()

def train_one_epoch(model, optimizer, data_loader, validation_loader, device, epoch):
    model.train()
    running_loss = 0.0
    prog_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

    for images, targets in prog_bar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        prog_bar.set_postfix(loss=running_loss/len(data_loader), epoch=epoch+1)

    avg_loss = running_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Validate the model after each epoch
    avg_iou = validate_model(model, validation_loader, device)
    mAP = validate_model_with_map(model, validation_loader, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average IoU: {avg_iou:.4f}, mAP: {mAP:.4f}")

# Training and validation
start_time = time.time()
num_epochs = 3
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, val_loader, device, epoch)
    lr_scheduler.step()

elapsed_time = time.time() - start_time
print(f"Training completed in: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")