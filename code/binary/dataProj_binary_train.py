import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
import gc
from functools import partial
import torch
import torchvision.models as models
import albumentations as A  
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
import numpy as np
import json
from sklearn.model_selection import train_test_split, KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
import segmentation_models_pytorch as smp

from dataProj_binary_utils import (  
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    get_loadersTest,
    validation_loop,
    save_predictions_as_imgs,
)


gc.collect()
tqdm = partial(tqdm, position=0, leave=True)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
decayLR1 = 0.9
decayLR2 = 0.98
DEVICE = torch.device("cuda")
BATCH_SIZE_TRAIN = 128
BATCH_SIZE = 128
NUM_EPOCHS = 100   
KF = 0.1
NUM_WORKERS = 0
PIN_MEMORY = True
LOAD_MODEL = False
TEST = True
file_dir = ""
valid_loss = 1e10
currentCheckPoint = 'segmentation'


# Train function over the entire train set.
def train_fn(loader, model, optimizer, loss_fn): 
    model.train()
    loop = tqdm(loader)   
 
    epochTrain_loss = []
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE) 
        targets = targets.to(device=DEVICE) 

        predictions = model(data)

        loss = loss_fn(predictions, targets)

        epochTrain_loss.append(loss.item())
        
        # backward
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    length = len(epochTrain_loss)
    return sum(epochTrain_loss)/length


def main():

    global valid_loss, LEARNING_RATE
    
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(

            #sequence:      SAR        HAND               slope        
            mean= [-13.669496/255.0, 8.085045/255.0, 2.8911793/255.0],
            std =[5.99311/255.0, 20.154648/255.0, 5.018089/255.0],
            ),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Normalize(          
      
            mean= [-13.669496/255.0, 8.085045/255.0, 2.8911793/255.0],
            std =[5.99311/255.0, 20.154648/255.0, 5.018089/255.0],
            ),
        ],
    )

    # Using Unet with the ResNet34 as the encoder. 
    model = smp.Unet(encoder_name='resnet34', in_channels=3, classes=1)
    
    model = model.to(DEVICE)
    model = torch.nn.DataParallel(model)

    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(decayLR1, decayLR2))

    # Adjust the learning rate when the loss stops decreasing for 5 epochs. 
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 5, threshold=1e-4, factor=0.9)

    
    if TEST:
        print('test...')
        load_checkpoint(torch.load(f"dataPrj_checkpoints/{currentCheckPoint}.pth.tar"), model)
        
        testList = []
        with open('test_cor.csv') as input_test:

            test_reader = csv.reader(input_test)
            for row in test_reader:
                testList += row  

            print('test len: ', len(testList))
            
            test_loader = get_loadersTest(
                file_dir,
                testList,
                BATCH_SIZE,
                val_transforms,
                NUM_WORKERS,
                PIN_MEMORY,
            )      
            
    
            save_predictions_as_imgs(test_loader, model, loss_fn, DEVICE, BATCH_SIZE, currentCheckPoint)
        
        
        return

    elif LOAD_MODEL:

        ## Enable this if you want to train with existing weights obtained previously. 
        # print('load model for new train')
        # load_checkpoint(torch.load(f"dataPrj_checkpoints/{currentCheckPoint}.pth.tar"), model)
    
    
    tloss = []
    vloss = []
    in_tv = []  

    with open('train_cor.csv') as inData:
        reader = csv.reader(inData)
        for row in reader:
            in_tv+=row   
    
    trainIDX, valiIDX = train_test_split(list(range(len(in_tv))), test_size = KF, random_state=42)
    train_list = [in_tv[i] for i in trainIDX]
    vali_list = [in_tv[i] for i in valiIDX]

    for epoch in range(NUM_EPOCHS):
        torch.cuda.empty_cache()
        print("current epoch", epoch)
        train_loader, val_loader = get_loaders(
            file_dir,
            train_list, 
            vali_list, 
            BATCH_SIZE_TRAIN,
            BATCH_SIZE,
            train_transform,
            val_transforms,
            NUM_WORKERS,
            PIN_MEMORY,
        )        

        eloss= train_fn(train_loader, model, optimizer, loss_fn)
        print('loss: ', eloss)
        tloss.append(eloss)  

        loss = validation_loop(val_loader, model, loss_fn)
        vloss.append(loss)
        
        scheduler.step(loss)
        
        print(optimizer.state_dict()['param_groups'][0]['lr'])

        if loss < valid_loss:
            valid_loss = loss
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, epoch)
            print("new validation loss: ", valid_loss)
      


    with open("dataPrj_files/vLoss_{currentCheckPoint}.txt", "w") as vLoss:
        json.dump(vloss, vLoss)
    with open("dataPrj_files/tLoss_{currentCheckPoint}.txt", "w") as tLoss:
        json.dump(tloss, tLoss)
      
if __name__ == "__main__":
    main()
