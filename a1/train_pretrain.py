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

# def accuracy(outputs, labels):
#     _, preds = torch.max(outputs, dim=1) 
#     return torch.tensor(torch.sum(preds == labels).item() / len(preds)), preds

def train(model, train_loader, criterion, optimizer, epoch, writer):
    model.train()
    steps = len(train_loader)
    train_loss = 0.0
    train_acc = 0.0
    
    for batch_id, (imgs, trgt) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        
        imgs = imgs.to(device)
        trgt = trgt.to(device)
        
        preds = model(imgs)
        loss = criterion(preds, trgt)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # _, outputs = torch.max(preds, dim=1)
        train_acc+= accuracy(preds, trgt)
        
        # print freq = 50
        if (batch_id+1) %50==0:
            running_loss = train_loss / batch_id
            running_acc = train_acc / batch_id
            print(f'[TRAIN] epoch-{epoch:0{len(str(epoch))}}/50,'
                    f'batch-{batch_id + 1:0{len(str(steps))}}/{steps},'
                    f'[LOSS]-{running_loss:.3f}, [ACC]-{running_acc:.3f}')
            
            # summary writing
            total_step = (epoch - 1) * len(train_loader) + batch_id + 1
            info = {
                'loss': running_loss,
                'acc': running_acc,
            }
            
            writer.add_scalars('train', info, total_step)
            sys.stdout.flush()
        
        
    return train_loss/len(train_loader)

def validate(model, val_loader, criterion, epoch, writer):
    model.eval()
    steps = len(val_loader)
    
    val_loss = 0.0
    val_acc = 0.0
    
    with torch.no_grad():
        for batch_id, (imgs, trgt) in tqdm(enumerate(val_loader)):
            imgs = imgs.to(device)
            trgt = trgt.to(device)
            
            preds = model(imgs)
            loss = criterion(preds, trgt)
            
            val_loss += loss.item()
            
            val_acc += accuracy(preds, trgt)
            
            # print freq = 100
            if (batch_id+1) %100==0:
                running_loss = val_loss / batch_id
                running_acc = val_acc / batch_id
                print(f'[VAL] epoch-{epoch:0{len(str(epoch))}}/20,'
                        f'batch-{batch_id + 1:0{len(str(steps))}}/{steps},'
                        f'[LOSS]-{running_loss:.3f}, [ACC]-{running_acc:.3f}')
                
                # summary writing
                total_step = (epoch - 1) * len(train_loader) + batch_id + 1
                info = {
                    'loss': running_loss,
                    'acc': running_acc,
                }
                
                writer.add_scalars('val', info, total_step)
                sys.stdout.flush()
            
        print(f'[VAL]: Loss: {val_loss/len(val_loader)}, Acc: {val_acc/len(val_loader)}')
        return val_loss/len(val_loader)
    
    


if __name__ == '__main__':
    
    # Set random seed for reproducibility
    init_seeds(4673)
    
    data_root_path = "/home/akash/spring23_coursework/cap5516/a1/chest_xray"
    
    # Load datasets
    train_dataset = ImageFolder(osp.join(data_root_path, 'train'), transform=transform_data(mode='train'))
    val_dataset = ImageFolder(osp.join(data_root_path, 'val'), transform=transform_data(mode='val'))
    test_dataset = ImageFolder(osp.join(data_root_path, 'test'), transform=transform_data(mode='test'))
    
    print("Dataset classes:", train_dataset.classes)
    print("Labels value:", train_dataset.class_to_idx)
    
    # HYPERPARAMS
    BATCH_SIZE = 48
    LR = 0.001
    N_EPOCHS = 20
    
    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    # Define model
    model = torchvision.models.resnet50(pretrained=True).to(device)
    
    # for name, params in model.named_parameters():
    #     print(name)
        
    # # modify fc layer
    model.fc = nn.Linear(2048, 2).to(device) # number of classes -2 
    
    # define criterion
    criterion = nn.CrossEntropyLoss()
    
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0, eps=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-7, patience=5, factor=0.1,
                                                     verbose=True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 45], gamma=0.1, verbose=True)

    # exp_id = args.exp_id
    exp_id = 'ft_all'
    save_path = os.path.join('./train_log_wts', exp_id)
    model_save_dir = os.path.join(save_path, time.strftime('%m-%d-%H-%M'))
    writer = SummaryWriter(model_save_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # # WANDB RUN NAME DECLARATION
    # wandb.run.name = args.exp_id

    prev_best_val_loss = 10000
    prev_best_train_loss = 10000
    prev_wt_cons_loss = 10000
    prev_best_val_loss_model_path = None
    prev_best_train_loss_model_path = None
    
    # wandb.watch(model)
    # wandb.watch(ema_model)

    for e in tqdm(range(1, N_EPOCHS + 1)):
        train_loss = train(model, train_loader, criterion, optimizer, e, writer)
        val_loss = validate(model, val_loader, criterion, e, writer)
        
        if val_loss < prev_best_val_loss:
            print("Yay!!! Got the val loss down...")
            val_model_path = os.path.join(model_save_dir, f'best_model_val_loss_{e}.pth')
            torch.save(model.state_dict(), val_model_path)
            prev_best_val_loss = val_loss
            if prev_best_val_loss_model_path and e < 25:
                os.remove(prev_best_val_loss_model_path)
            prev_best_val_loss_model_path = val_model_path

        if train_loss < prev_best_train_loss:
            print("Yay!!! Got the train loss down...")
            train_model_path = os.path.join(model_save_dir, f'best_model_train_loss_{e}.pth')
            torch.save(model.state_dict(), train_model_path)
            prev_best_train_loss = train_loss
            if prev_best_train_loss_model_path and e<25:
            # if prev_best_train_loss_model_path:
                os.remove(prev_best_train_loss_model_path)
            prev_best_train_loss_model_path = train_model_path
        scheduler.step(val_loss)
        
    
    
    
    