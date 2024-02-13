import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,5,6"
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

from utils_dataProj import (  
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    get_loadersTest,
    validation_loop,
    save_predictions_as_imgs,
    NSE_Loss,
)


gc.collect()
tqdm = partial(tqdm, position=0, leave=True)

# Hyperparameters etc.
LEARNING_RATE = 5e-4
decayLR1 = 0.9
decayLR2 = 0.98
DEVICE = torch.device("cuda")
BATCH_SIZE_TRAIN = 64
BATCH_SIZE = 64
NUM_EPOCHS = 80   
KF = 0.1
NUM_WORKERS = 0 
PIN_MEMORY = True
LOAD_MODEL = False
TEST = True
file_dir = ""
valid_loss = 1e10



def train_fn(loader, model, optimizer, loss_fn): 
    model.train()
    loop = tqdm(loader)   
 
    epochTrain_loss = []
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE) 
        targets = targets.to(device=DEVICE) 

        predictions = model(data)
        
        loss1 = loss_fn[0](predictions, targets)
        loss3 = loss_fn[1](predictions, targets)

        loss = 0.5*loss1+0.5*loss3

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

            #sequence:      HAND               landcover         precip1         precip2-1         precip3-2      precip5-3          precip7-5           sm1            sm2            prev_SAR    
            mean= [8.085045/255.0,0.303/255.0, 0.214/255.0, 0.113/255.0, 0.503/255.0, 11.097879/255.0, 6.300379/255.0, 4.5807447/255.0, 10.080237/255.0, 10.893396/255.0, 20.901936/255.0, 20.849417/255.0, -13.641111/255.0],
            std =[20.154648/255.0,0.46/255.0, 0.41/255.0, 0.317/255.0, 0.5/255.0, 19.518549/255.0, 10.485548/255.0, 11.979355/255.0, 12.549833/255.0, 20.573872/255.0, 3.0335426/255.0, 3.2292476/255.0, 5.7359266/255.0],
            ),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Normalize(          
            mean= [8.085045/255.0,0.303/255.0, 0.214/255.0, 0.113/255.0, 0.503/255.0, 11.097879/255.0, 6.300379/255.0, 4.5807447/255.0, 10.080237/255.0, 10.893396/255.0, 20.901936/255.0, 20.849417/255.0, -13.641111/255.0],
            std =[20.154648/255.0,0.46/255.0, 0.41/255.0, 0.317/255.0, 0.5/255.0, 19.518549/255.0, 10.485548/255.0, 11.979355/255.0, 12.549833/255.0, 20.573872/255.0, 3.0335426/255.0, 3.2292476/255.0, 5.7359266/255.0],
            ),
        ],
    )

    # switch between different models
    #model = smp.Unet(encoder_name='resnet50', in_channels=13, classes=1)
    #model = smp.MAnet(encoder_name='resnet50', in_channels=13, classes=1)
    #model = smp.PSPNet(encoder_name='resnet50', in_channels=13, classes=1)
    model = smp.DeepLabV3Plus(encoder_name='resnet50', in_channels=13, classes=1)
    
    model = model.to(DEVICE)
    model = torch.nn.DataParallel(model)

    loss_fn3 = nn.L1Loss()      
    loss_fn1 = NSE_Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(decayLR1, decayLR2))
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 5, threshold=1e-4, factor=0.9)

    
    if TEST:
        # load saved checkpoints to investigate trained model on test set
        print('test...')
        currentCheckPoint = 'DeepLabV3P'
        load_checkpoint(torch.load(f"dataPrj_checkpoints/{currentCheckPoint}.pth.tar"), model)
        
        testList = []
        with open('test_cor.csv') as input_test:

            test_reader = csv.reader(input_test)
            for row in test_reader:
                testList += row  
            
            test_loader = get_loadersTest(
                file_dir,
                testList,
                BATCH_SIZE,
                val_transforms,
                NUM_WORKERS,
                PIN_MEMORY,
            )      
            
    
            save_predictions_as_imgs(test_loader, model, [loss_fn1,loss_fn3], DEVICE, BATCH_SIZE, currentCheckPoint)
        
        return

    elif LOAD_MODEL:
        print('load model for new train')
        load_checkpoint(torch.load(f"dataPrj_checkpoints/{currentCheckPoint}.pth.tar"), model)
    
    tloss = []
    vloss = []
    in_tv = []  

    with open('train_cor.csv') as inData:
        reader = csv.reader(inData)
        for row in reader:
            in_tv+=row   
    
    # get train and validation sets 
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

        eloss= train_fn(train_loader, model, optimizer, [loss_fn1, loss_fn3])
        print('loss: ', eloss)
        tloss.append(eloss)  

        loss = validation_loop(val_loader, model, [loss_fn1, loss_fn3])
        vloss.append(loss)
        
        scheduler.step(loss)
        
        print(optimizer.state_dict()['param_groups'][0]['lr'])

        # save current status of the model if a better loss on the validation set is achieved
        if loss < valid_loss:
            valid_loss = loss
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, epoch)
            print("new validation loss: ", valid_loss)
      

    with open("dataPrj_files/vLoss_DeepLabV3P.txt", "w") as vLoss:
        json.dump(vloss, vLoss)
    with open("dataPrj_files/tLoss_DeepLabV3P.txt", "w") as tLoss:
        json.dump(tloss, tLoss)
      
if __name__ == "__main__":
    main()
