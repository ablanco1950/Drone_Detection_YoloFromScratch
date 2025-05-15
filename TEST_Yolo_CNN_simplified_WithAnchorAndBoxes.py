# https://medium.com/@whyamit404/how-to-implement-a-yolo-object-detector-from-scratch-in-pytorch-e310829d92e6
# article on anchor boxes
# https://dkharazi.github.io/notes/ml/cnn/yolo
#
# https://stackoverflow.com/questions/64345584/how-to-properly-use-cv2-findcontours-on-opencv-version-4-4-0
# https://sites.google.com/educarex.es/python/librer%C3%ADas/opencv/contornos-de-objetos

GRID_SIZE=32
IMG_SIZE=640
CONF_THRESHOLD=3.5e-18 # PARA Cont10YoloCNN_epoch420250501.pth
CONF_ANCHOR_THRESHOLD = 0.5

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import cv2
import os
import re 

from GetAnchor import GetFit

# MODEL

class YOLO(nn.Module):
   
    def __init__(self, num_classes=1, num_anchors=1, grid_size=16):    
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
        # # Sobrea la salida de yolo ver la primera imagen en
        # https://medium.com/@chnwsw01/yolo-algorithm-c779b9b2018b
        self.detector = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Flatten(),
            #nn.Linear(512 * (grid_size // 4)**2, grid_size * grid_size * (num_anchors * 5 + num_classes)),
            nn.Linear(1024 * (grid_size // 4)**2, grid_size * grid_size * (num_anchors * 5 + num_classes)),
        )

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.detector(features)
        return predictions.view(-1, self.grid_size, self.grid_size, self.num_anchors * 5 + self.num_classes)


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
    #def __init__(self, grid_size=7, num_classes=20, num_anchors=3):
    #def __init__(self, grid_size=7, num_classes=1, num_anchors=3):
    def __init__(self, grid_size=16, num_classes=1, num_anchors=1):     
        super(YOLO, self).__init__()
        self.backbone = YOLOBackbone()
        self.head = YOLOHead(grid_size, num_classes, num_anchors)

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions


class YOLODataset(Dataset):
    #def __init__(self, img_dir, label_dir, transforms=None):
    def __init__(self, img_dir,  transforms=None):    
        self.img_dir = img_dir
        #self.label_dir = label_dir
        self.transforms = transforms
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        #label_path = os.path.join(self.label_dir, self.images[idx].replace(".jpg", ".txt"))

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640,640), interpolation = cv2.INTER_AREA) # adapting any size image
        # Load annotations
        boxes = []
       
        if self.transforms:
            image = self.transforms(image)
        #return image, torch.tensor(boxes)
        return image
    

"""

Inference Function

Here’s how you can perform inference using the trained YOLO model:
"""
# Get indx of positions of grids with its predictions
def perform_inference(model, image, CONF_THRESHOLD, TabPredGrid):
    """Runs inference on an image and returns bounding boxes."""
    model.eval()
    
    image_tensor=image
    predictions = model(image_tensor)
   
    boxes = []
    
    Otras_final_boxes=[]
    
    conf_threshold=CONF_THRESHOLD

   
    boxes= []
    Tab_i=[]
    Tab_j=[]
    
    for pred in predictions:
        
        
        for i in range (len(pred)):
            #print(len(pred[0]))
           
            for j in range (len(pred[i])):

                  conf_pred=pred[..., 0:5][i][j][4].detach().cpu().numpy()
                  
                  if conf_pred > conf_threshold:
                     
                         #print("Pasa conf_pred=" + str(conf_pred))
                           
                         boxes.append(pred[..., 0:5][i][j].detach().cpu().numpy()) # convert from tensor to  array
                                                                                   # for more simple operations
                         
                         Tab_i.append(i)
                         Tab_j.append(j)
                         TabPredGrid [i,j]=conf_pred
                         
                        
                  else:
                       continue
                  


    return boxes, Tab_i, Tab_j, TabPredGrid 


def load_pretrained_weights(model, weight_path):
    """Loads pre-trained weights into the YOLO model."""
    state_dict = torch.load(weight_path)
    #model.load_state_dict(state_dict, strict=False)  # strict=False allows partial loading
    model.load_state_dict(state_dict)  
    print("Pre-trained weights loaded successfully!")
    return model
   
def loadimages(dirname):
 
     imgpath = dirname + "\\"
      
     
     images = []
     TabFileName=[]
   
    
     print("Reading images from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
        
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff|JPEG)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                
                 
                 image = cv2.imread(filepath)
                 image = cv2.resize(image, (640,640), interpolation = cv2.INTER_AREA) # adapting any size image
                 #print(filepath)
                 #print(image.shape)                           
                 images.append(image)
                 TabFileName.append(filename)
                 
                 Cont+=1
                 #if Cont > 1:break
     print("Readed " + str(len(images)))
     #cv2.imshow('True', images[0])
     #cv2.waitKey(0)
   
     return images, TabFileName

#
# Receives: Xmin , Ymin, Xmax, Ymax that delimits the box
# TabPredGrid array with the scores predicted for each grid of the box
# CONF_THRESHOLD a factor to fits the score of each grid betwin 0-3
# Returns the conf assigned to the box


def CalcPred( Xmin , Ymin, Xmax, Ymax, TabPredGrid, CONF_THRESHOLD):
    TotalConf=0.0
    ContGrid=0

    #print("len(TabPredGrid) "+ str(len(TabPredGrid)))
    #print("len(TabPredGrid[0]) "+ str(len(TabPredGrid[0])))

    # calaculate the position in terms of grids
    GridXmin= int(Xmin/GRID_SIZE)
    GridXmax= int(Xmax/GRID_SIZE)
    GridYmin= int(Ymin/GRID_SIZE)
    GridYmax= int(Ymax/GRID_SIZE)
    
    # calculate the mean of conf of all grids
    for i in range(len(TabPredGrid)):
        if i< GridXmin or i > GridXmax: continue
        for j in range(len(TabPredGrid[0])):
            if j< GridYmin or j > GridYmax: continue
            TotalConf=TotalConf+ (TabPredGrid[i][j] / CONF_THRESHOLD)
            ContGrid=ContGrid+1
            
    return   TotalConf/ContGrid      
            


# MAIN

import matplotlib.pyplot as plt
import matplotlib.patches as patches

dirnameImages="test\\images"
#dirnameImages="Test1"
TabImages, TabfileName = loadimages(dirnameImages)

import numpy as np

weight_path = "Cont10YoloCNN_epoch420250501.pth" # ESTE S EL BUENO

model = YOLO(grid_size=7, num_classes=1, num_anchors=1)
model = load_pretrained_weights(model, weight_path)
model.eval()

# Freeze backbone layers
for param in model.backbone.parameters():
    param.requires_grad = False

# Example: Initialize DataLoader
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

test_dataset = YOLODataset(img_dir="test/images", transforms=ToTensor())
#test_dataset = YOLODataset(img_dir="Test1", transforms=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

indexImg=0

# array with the predictions for each grid
TabPredGrid=np.zeros([int(IMG_SIZE/GRID_SIZE),int(IMG_SIZE/GRID_SIZE)])

for img in test_loader:       

        
        boxes, Tab_i, Tab_j, TabPredGrid = perform_inference(model, img, CONF_THRESHOLD,TabPredGrid)

        image=TabImages[indexImg]
        indexImg=indexImg+1

        Img_size=IMG_SIZE
        Grid_size=GRID_SIZE
        
        image2=image.copy()
        img_rgb=image.copy()
        
        for i in range (len(Tab_i)):

            Xmin= Tab_j[i] * Grid_size
            Ymin=Tab_i[i] * Grid_size
            Xmax=Xmin+ Grid_size
            Ymax=Ymin+Grid_size

           
            if Xmin==0: continue
            if Xmax == IMG_SIZE: continue
            if  Ymin==0: continue
            if Ymax == IMG_SIZE: continue
            
            # fill grids detecteds with black color to patched the image
            # https://stackoverflow.com/questions/52509316/opencv-rectangle-filled answer 17
            image = cv2.rectangle(image, (Xmin, Ymin), (Xmax, Ymax),(0,0,0), -1)

        
        #cv2.imshow("Patched",image)
        #cv2.waitKey(0)

        
        
        ContDrones=0
        # https://sites.google.com/educarex.es/python/librer%C3%ADas/opencv/contornos-de-objetos
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        thresh, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV)
        #cv2.imshow("binary",binary)
        #cv2.waitKey(0) 
                
        contornos, jerarquia = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # BUENA

        # ordenar contornos por area (not aplied)
        # https://stackoverflow.com/questions/32669415/opencv-ordering-a-contours-by-area-python

        for contour in contornos:
            Area=cv2.contourArea(contour)
            # limit Area of contours to one grid          
            if   Area <= 32*32: continue
            #print(Area) 
        # Find Perimeter of contour and it should be a closed contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)# queda con muchos picos
            
            
            if len(approx) < 4:
                continue
            else :
                    #contour_with_license_plate = approx
                    x, y, w, h = cv2.boundingRect(contour)

                                       
                    #cv2.rectangle(image2, (x, y), (x+w, y+h), color=(0, 0, 255),thickness=2)
                    
                    # Use anchors to fiiter boxes
                    confAnchor, w, h=GetFit(w,h) # se desecha conf por similitud con anchors # SOLO EN CASO de conf por ajuste a anchor
                    confAnchor=confAnchor.detach().cpu().numpy()
                    confAnchor=confAnchor[0]

                    conf = CalcPred( x , y, x+w, y+w, TabPredGrid, CONF_THRESHOLD)
                    
                    #print("CONF "+ str(conf))
                    #print("CONFAnchor "+ str(confAnchor))                    
                    
                    if confAnchor < CONF_ANCHOR_THRESHOLD and conf > 0.0:
                      
                      cv2.rectangle(image2, (x, y), (x+w, y+h), color=(255, 0, 0),thickness=2)
                      #confText=str(1-conf) # SOLO EN CASO de conf por ajuste a anchor
                      confText=str(conf)
                      confText=confText[0:4]
                      confText="drone" + confText
                      cv2.putText(image2, confText, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
                    
                      ContDrones= ContDrones+1
                 
        #cv2.imshow("Patched",image2)
        #cv2.waitKey(0)
        print(" Drones " +str(ContDrones))    

        fig, axs = plt.subplots(1,3, figsize=(15,5))
        axs[0].imshow(img_rgb);      axs[0].set_title('Original');    axs[0].axis('off')
        axs[1].imshow(image); axs[1].set_title('Patched'); axs[1].axis('off')
        axs[2].imshow(image2);     axs[2].set_title('Boxes'); axs[2].axis('off')
        plt.tight_layout(); plt.show()
