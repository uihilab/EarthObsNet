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
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR, ChainedScheduler
from dataPrj_SS_model import MAnet_SS
from dataPrj_SS_utils import (  
    load_checkpoint,
    save_checkpoint,
    get_loadersTrain,
    get_loadersVali_Test,
    validation_loop,
    save_predictions_as_imgs,
    NSE_Loss,
)



gc.collect()
tqdm = partial(tqdm, position=0, leave=True)

# Hyperparameters etc.
LEARNING_RATE = 5e-3
decayLR1 = 0.9
decayLR2 = 0.98
DEVICE = torch.device("cuda")
BATCH_SIZE = 64
BATCH_SIZE_TEST = 64
NUM_EPOCHS = 150
KF = 0.1
LOAD_MODEL = False
TEST = True
valid_loss = 1e10
currentCheckPoint = 'SS_150'



def csvRead(fileName):
    in_tv = []
    with open(fileName) as inData:
        reader = csv.reader(inData)
        for row in reader:
            in_tv+=row 
    return in_tv  


def train_fn(loader, model, optimizer, loss_fn): 
    model.train()
    
    loop = tqdm(loader)   
 
    epochTrain_loss = []
    
    for batch_idx, (outInput, data, targets) in enumerate(loop):
        outInput = outInput.to(device=DEVICE) 
        data = data.to(device=DEVICE) 
        targets = targets.to(device=DEVICE) 

        #forward
        predictions = model(outInput, data)
        
        loss1 = loss_fn[0](predictions, targets)
        loss3 = loss_fn[1](predictions, targets)
        loss = 0.5*loss1+0.5*loss3

        epochTrain_loss.append(loss.item())

        
        # backward
        optimizer.zero_grad()
        loss.backward()

        # update model params
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    length = len(epochTrain_loss)
    return sum(epochTrain_loss)/length


def main():

    global valid_loss

    
    model = MAnet_SS(13,1)
    model = model.to(DEVICE)
    model = torch.nn.DataParallel(model)


    loss_fn3 = nn.L1Loss()      #this is the 3rd loss
    loss_fn1 = NSE_Loss()

    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(decayLR1, decayLR2))

    # set up both the learning rate warm-up and reduce-on-plateau
    scheduler1 = LinearLR(optimizer, start_factor = 0.02, end_factor = 1.0, total_iters=50)
    scheduler2 = ReduceLROnPlateau(optimizer, 'min', patience = 5, threshold=1e-4, factor=0.9)
    
  
    if TEST:
        print('test...')         
        load_checkpoint(torch.load(f"dataPrj_checkpoints/mergeInput/{currentCheckPoint}.pth.tar"), model)


        testCurrentList = csvRead('test.csv')
        testOutsideList = None if 'S0' in currentCheckPoint else csvRead('testOutside.csv')

        print('test len check: ', len(testCurrentList))

        test_loader = get_loadersVali_Test(
            testCurrentList,
            testOutsideList,
            BATCH_SIZE_TEST,
        )   

        save_predictions_as_imgs(test_loader, model, [loss_fn1,loss_fn3], DEVICE, BATCH_SIZE_TEST, currentCheckPoint)
        
        return

    elif LOAD_MODEL:
        print('load model for new train')
        load_checkpoint(torch.load(f"dataPrj/mergeInput/{currentCheckPoint}.pth.tar"), model)



    tloss = []
    vloss = []
    
    currentLis = csvRead('train.csv')

    # get data from neighboring areas for SA and SS
    outsideLis = None if 'S0'in currentCheckPoint else csvRead('trainOutside.csv')  

    
    trainIDX, valiIDX = train_test_split(list(range(len(currentLis))), test_size = KF, random_state=42)
    train_list = [currentLis[i] for i in trainIDX]
    vali_list = [currentLis[i] for i in valiIDX]

    # get data from neighboring areas for SA and SS
    train_list_outside = None if not outsideLis else [outsideLis[i] for i in trainIDX]
    vali_list_outside = None if not outsideLis else [outsideLis[i] for i in valiIDX]


    for epoch in range(NUM_EPOCHS):
        
        torch.cuda.empty_cache()
        print("current epoch", epoch)
        
        train_loader = get_loadersTrain(
            train_list,
            train_list_outside, 
            BATCH_SIZE,
        )  
        

        val_loader = get_loadersVali_Test(
            vali_list,                
            vali_list_outside,
            BATCH_SIZE,
        )
        

        eloss= train_fn(train_loader, model, optimizer, [loss_fn1, loss_fn3])
        print('loss: ', eloss)
        tloss.append(eloss)  
        
        
        
        loss = validation_loop(val_loader, model, [loss_fn1, loss_fn3])
        vloss.append(loss)
        

        # The two schedulers will both fire for the first 50 iterations, but in our case 
        # the latter wouldn't really take any actions after several dozens of more 
        # iterations after the former stopped working, as it is conditioned on loss. Therefore, they work perfectly in our case.
        # If your schedulers will mess up with each other and you don't want that, you can implement 
        # your schedulers in differents ways to avoid the unfavorable consequences. 
        scheduler1.step()
        scheduler2.step(loss)
        
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
        
      

        with open(f"dataPrj_files/mergeInput/vLoss_{currentCheckPoint}.txt", "w") as vLoss:
            json.dump(vloss, vLoss)
        with open(f"dataPrj_files/mergeInput/tLoss_{currentCheckPoint}.txt", "w") as tLoss:
            json.dump(tloss, tLoss)
      
if __name__ == "__main__":
    main()
