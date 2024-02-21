# Basic and ML imports
import os
import numpy as np
import pandas as pd
from skimage import io
from skimage.transform import resize
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# For Mask R-CNN
import torchvision.models.detection.mask_rcnn

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
        target["boxes"] = torch.as_tensor([bbox], dtype=torch.float32) if bbox is not None else torch.zeros((0, 4), dtype=torch.float32)
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
df = pd.read_csv('/scratch/hmudigon/Experimentation/datasets/good/TBX11K/data.csv')
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['target'])

# Assuming images are stored in 'train/' for both train and validation
train_dataset = CustomDataset(dataframe=train_df, img_dir='/scratch/hmudigon/Experimentation/datasets/good/TBX11K/train/', transform=transform)
val_dataset = CustomDataset(dataframe=val_df, img_dir='/scratch/hmudigon/Experimentation/datasets/good/TBX11K/test/', transform=transform)

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

import torchvision

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
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask,
                                                                                               hidden_layer,
                                                                                               num_classes)
    
    return model

# Adjust num_classes to match your dataset (Background + number of classes)
num_classes = 2
model = get_model_instance_segmentation(num_classes)
print(model)
