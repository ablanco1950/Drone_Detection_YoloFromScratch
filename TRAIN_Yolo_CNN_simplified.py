
# CON UN GRID SIZE = IMAGE SIZE
# SE EQUIPARA YOLO A UNA REGRESION, NO ES ASI
# https://medium.com/@whyamit404/how-to-implement-a-yolo-object-detector-from-scratch-in-pytorch-e310829d92e6
# https://dkharazi.github.io/notes/ml/cnn/yolo
# https://discuss.pytorch.org/t/using-sigmoid-function-in-cnn-classifier/163712/4
# https://medium.com/towards-data-science/binary-image-classification-in-pytorch-5adf64f8c781
# https://stackoverflow.com/questions/50575301/yolo-object-detection-how-does-the-algorithm-predict-bounding-boxes-larger-than

IMG_SIZE = 640
GRID_SIZE = 7
import torch
import torch.nn as nn

class YOLO(nn.Module):
    
    def __init__(self, num_classes=1, num_anchors=1, grid_size=GRID_SIZE):
        super(YOLO, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.grid_size = grid_size

        # Backbone: Feature extractor (e.g., simplified CNN for demonstration)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
        )

        # Detection Head: Outputs bounding boxes, confidence scores, and class probabilities
        self.detector = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Flatten(),
           
            nn.Linear(1024*100 , grid_size * grid_size * (num_anchors * 5 + 1)),
        )

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.detector(features)
        
        #return predictions.view(-1, self.grid_size, self.grid_size, self.num_anchors * 5 + self.num_classes)
        return predictions.view(-1, 1, 1, self.num_anchors * 5 + self.num_classes)

# Instantiate the model

model = YOLO(num_classes=1)
print(model)

"""
Defining the Network
The backbone of YOLO is a convolutional neural network (CNN), with layers for feature extraction,
followed by a detection head for predictions.
YOLO’s magic lies in its modularity, so we’ll build it step by step.

Step 1: Modular Building Blocks

Instead of hardcoding the entire architecture, we’ll create reusable modules for convolutional layers with batch normalization
# and activation functions. This approach reduces redundancy and enhances clarity.
"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """A block of Conv2D -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

"""
Step 2: Building the Backbone

Now, let’s stack these ConvBlocks to form a lightweight backbone.
This backbone extracts meaningful features from the input image.
"""

class YOLOBackbone(nn.Module):
    def __init__(self):
        super(YOLOBackbone, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.layers(x)

"""
Step 3: Detection Head

The detection head predicts bounding boxes, confidence scores, and class probabilities.
Each prediction corresponds to a cell in the YOLO grid.

"""

class YOLOHead(nn.Module):
    def __init__(self, grid_size, num_classes, num_anchors):
        super(YOLOHead, self).__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.detector = nn.Conv2d(512, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        return self.detector(x).permute(0, 2, 3, 1).contiguous()

"""
Step 4: Assembling the YOLO Model

Let’s combine the backbone and detection head into a complete YOLO model.

"""

class YOLO(nn.Module):
    
    def __init__(self, grid_size=GRID_SIZE, num_classes=1, num_anchors=1):     
        super(YOLO, self).__init__()
        self.backbone = YOLOBackbone()
        self.head = YOLOHead(grid_size, num_classes, num_anchors)

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions

# Example usage

model = YOLO(grid_size=GRID_SIZE, num_classes=1, num_anchors=1)
print(model)


import numpy as np


import torchvision.transforms as transforms


# COPIED FROM  https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/models/losses.py#L10C51-L10C51
def IOULoss(pred, target):
        reduction="none"
        loss_type="iou"
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if loss_type == "iou":
            loss = 1 - iou ** 2
        elif loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss
"""
6. Loss Function
“If training is the journey, the loss function is your compass.”

YOLO uses a multi-component loss function:

Localization Loss: Penalizes inaccurate bounding box predictions.
Confidence Loss: Measures how confident the model is about the presence of an object.
Classification Loss: Compares predicted classes with ground truth.
PyTorch Implementation

Here’s a PyTorch implementation of the YOLO loss function:
"""

def yolo_loss(predictions, targets, num_classes, lambda_coord=5, lambda_noobj=0.5):
    """
    Computes YOLO loss.
    - predictions: Predicted tensor.
    - targets: Ground truth tensor.
    """
    
   

    
    # Unpack predictions and targets
    pred_boxes = predictions[..., :4]

    #print(len(predictions[0][0]))
    #print(len(targets[0][0]))
    #print(len(pred_boxes[0][0]))
    #print(len(targets[0]))
    
    #pred_conf = predictions[..., 4] 
    
    pred_conf = predictions[..., 5]   

    
    #print (pred_conf)
    #print(pred_conf)
    #print(pred_conf[0][0][0])
    #print(ppa)
                                      
    pred_classes = predictions[..., 5:]


    #target_boxes = targets[..., :4]  # Corregido
    #print(targets[0][0])
   
    target_boxes = targets[..., 1:5]
    
   
    
    box_loss = 0.01
    
    for i in range (len(pred_boxes[0])):
         for j in range(len(pred_boxes[0][0])):
              
              box_loss = box_loss+IOULoss(pred_boxes[0][i][j], target_boxes[0][0])
              
    
    
   
    total_loss = box_loss 
    return total_loss

"""
“Every masterpiece starts with a single stroke.” Now you’ve built the foundation of the YOLO model, prepared your data, and defined the loss function. Next, we’ll tackle training and evaluation!

7. Training the YOLO Detector
“Training a YOLO model is like tuning an orchestra. Every component must work in harmony to detect objects accurately.”

DataLoader

To train the YOLO detector, you need a DataLoader that feeds the model images, bounding boxes, and labels in batches.

If you’ve followed the dataset preparation steps, your annotations should already be in YOLO format.

Custom Dataset Class

Here’s a custom PyTorch dataset class to handle images and annotations:
"""

import torch
from torch.utils.data import Dataset
import cv2
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace(".jpg", ".txt"))

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        boxes = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_label, x, y, w, h = map(float, line.strip().split())
                boxes.append([class_label, x, y, w, h])

        if self.transforms:
            image = self.transforms(image)
        return image, torch.tensor(boxes)

def load_pretrained_weights(model, weight_path):
    """Loads pre-trained weights into the YOLO model."""
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict, strict=False)  # strict=False allows partial loading
    print("Pre-trained weights loaded successfully!")
    return model


    

# Example: Initialize DataLoader
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

train_dataset = YOLODataset(img_dir="valid/images", label_dir="valid/labels", transforms=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

"""
Training Loop

“The training loop is where the magic happens. Every epoch brings your model closer to perfection.”

Here’s how you can implement the YOLO training loop.
"""

import torch.optim as optim

# Initialize model, optimizer, and loss function


weight_path = "Cont10YoloCNN_epoch100.pth"

model = YOLO(grid_size=GRID_SIZE, num_classes=1, num_anchors=1)

# To continue from a model trained before
#model = load_pretrained_weights(model, weight_path)



# Freeze backbone layers
for param in model.backbone.parameters():
    param.requires_grad = False

# Train only detection head
optimizer =torch.optim.Adagrad(model.head.parameters(), lr=0.005, lr_decay=0.0005, weight_decay=0.0005, initial_accumulator_value=0, eps=1e-10)

criterion = yolo_loss  # Your loss function from earlier

# Training loop
num_epochs = 200
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, targets in train_loader:
        
        # Forward pass
        predictions = model(images)

        
        
        # Loss calculation
       
        
        loss = criterion(predictions, targets, num_classes=1) # Esta es la loss usada pero nn.crossentropy no admite num_clases

        
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")    
    if (epoch+1) % 2 == 0:  
    # Print the epoch duration
      torch.save(model.state_dict(), f'Cont10YoloCNN_epoch{epoch+1}.pth', _use_new_zipfile_serialization=False)
   
torch.save(model.state_dict(), f'Cont10YoloCNN_epoch{epoch+1}.pth', _use_new_zipfile_serialization=False)    
    
