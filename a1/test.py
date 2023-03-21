import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
import random
import cv2
import copy
import os
import os.path as osp
from sklearn.model_selection import train_test_split
import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image

from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from tqdm import tqdm
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def init_seeds(seed=0, cuda_deterministic=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
        if cuda_deterministic:  # slower, more reproducible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:  # faster, less reproducible
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            
            
# Declare data augmentation transforms
def transform_data(mode):
    if mode=='train':
        img_transforms = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.RandomGrayscale(),
            T.ToTensor(),
            T.Normalize([0.489, 0.456, 0.406], [0.229, 0.224, 0.225])
            
        ])
    elif mode=='val':
        img_transforms = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.489, 0.456, 0.406], [0.229, 0.224, 0.225])
            
        ])
    else:
        img_transforms = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.489, 0.456, 0.406], [0.229, 0.224, 0.225])
            
        ])
        
    return img_transforms


def accuracy(preds, labels):
    preds = torch.exp(preds)
    top_p,top_class = preds.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))

def test(model, test_loader, criterion):
    model.eval()
    steps = len(test_loader)
    
    val_loss = 0.0
    val_acc = 0.0
    
    with torch.no_grad():
        for batch_id, (imgs, trgt) in tqdm(enumerate(test_loader)):
            imgs = imgs.to(device)
            trgt = trgt.to(device)
            
            preds = model(imgs)
            loss = criterion(preds, trgt)
            
            val_loss += loss.item()
            
            val_acc += accuracy(preds, trgt)
            
        print(f'[TEST]: Loss: {val_loss/len(test_loader)}, Acc: {val_acc/len(test_loader)}')
        return val_loss/len(test_loader)

if __name__ == '__main__':
    
    # Set random seed for reproducibility
    init_seeds(4673)
    
    data_root_path = "/home/akash/spring23_coursework/cap5516/a1/chest_xray"
    
    # Load datasets
    test_dataset = ImageFolder(osp.join(data_root_path, 'test'), transform=transform_data(mode='test'))
    
    # HYPERPARAMS
    BATCH_SIZE = 48
    
    # Dataloader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    # Define model
    model = torchvision.models.resnet50(pretrained=False).to(device)
    
    # # modify fc layer
    model.fc = nn.Linear(2048, 2).to(device) # number of classes -2 
    model.load_state_dict(torch.load('train_log_wts/scratch/03-21-00-05/best_model_val_loss_7.pth'), strict=True)
    # define criterion
    criterion = nn.CrossEntropyLoss()
    
    test_loss = test(model, test_loader, criterion)
        
        
    
    
    
    