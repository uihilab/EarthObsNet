import torch
import torch.nn as nn
import torchvision
from dataProj_binary_data import CarvanaDataset
from torch.utils.data import DataLoader
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import numpy as np
import math
from torch.autograd import Variable
import statistics
import json
import segmentation_models_pytorch as smp

checkpoint_dir = 'dataPrj_checkpoints'
filename = "segmentation"

def save_checkpoint(state, epoch):
    print("=> Saving checkpoint current epoch")
    checkpoint_path = checkpoint_dir+'/'+filename+'.pth.tar'
    torch.save(state, checkpoint_path)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    


# Data loaders for train and validation sets        
def get_loaders(
    file_dir,
    trainList,
    valiList,
    batch_size_train,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    
    train_ds = CarvanaDataset(
        f_dir=file_dir, l=trainList,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        f_dir=file_dir,l=valiList,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )


    return train_loader, val_loader

# Data loaders for test set 
def get_loadersTest(
    file_dir,
    testList,
    batch_size,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    
    test_ds = CarvanaDataset(
        f_dir=file_dir,l=testList,
        transform=val_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )


    return test_loader


def validation_loop(loader, model, loss_fn):

    model.eval()          
    torch.cuda.empty_cache()
    count = 0
    loss = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()
            outputs = model(x).cuda()
            target = y.cuda()

            outputs = outputs.detach().cpu()
            target = target.detach().cpu()            
            
            loss += loss_fn(outputs, target).item()

            count += 1

    loss /= count

    print("Validation Loss:", loss)   

    return loss  

    

def save_predictions_as_imgs(
    loader, model, criterion, device, batch, fileName='results'
):
    validation_loop(loader, model, criterion)
    model.eval()

    IoU, F1, A = [], [], []
    figure5List = [30, 44, 285, 247, 48, 328, 69, 167, 313, 383]  # A list of random IDs for which we will save some predicted flood maps for vasualization.
    header = ['NCOLS 256', 'NROWS 256', 'XLLCENTER 0', 'YLLCENTER 0', 'CELLSIZE 10', 'NODATA_VALUE -100']  # ASCII header to be attached to the file, so that it opens in GIS software.
    
    with torch.no_grad(): 

        for idx, (x, y) in enumerate(loader):
            output = model(x.to(device))
            target = y.to(device)
            
            for itemidx in range(target.shape[0]):

                iou, f1, a = getStastics(output.to(device)[itemidx], target[itemidx])
                IoU.append(iou)
                F1.append(f1)
                A.append(a)
                
                img_id = idx * 128 + itemidx
                #========================= 
                if img_id in figure5List:
                    
                    prediction_layer = output.to(device)[itemidx]
                    target_layer = target[itemidx]

                    prediction = torch.tanh(prediction_layer)
                    prediction = (prediction > 0).int()
                    
                    ## Save the predicted flood inundation map.
                    # with open(f'dataPrj_results/fig5/img/real_map_{img_id}.ASC', 'w') as file:
                    #     for line in header:
                    #         file.write(line+'\n')
                    # with open(f'dataPrj_results/fig5/img/real_map_{img_id}.ASC', 'a') as outfile:
                    #     out = prediction.detach().cpu().numpy().flatten()
                    #     outfile.write(' '.join(str(num) for num in out))
                        

        ## Save IoU, F1, and accuracy lists.
        # with open(f"dataPrj_files/IoU_prediction_{fileName}.txt", "w") as iou_file:
        #     json.dump(IoU, iou_file)   
        # with open(f"dataPrj_files/F1_prediction_{fileName}.txt", "w") as f1_file:
        #     json.dump(F1, f1_file)
        # with open(f"dataPrj_files/Accuracy_prediction_{fileName}.txt", "w") as a_file:
        #     json.dump(A, a_file)


        mean_median(IoU, F1, A)   


# Compute performance metrics. Three different metric points for each prediction-ground truth image pair.
def getStastics(predLayer, refLayer):

    predLayer.unsqueeze(0)
    refLayer.unsqueeze(0)

    prediction = torch.tanh(predLayer)
    prediction = (prediction > 0).float()
    tp, fp, fn, tn = smp.metrics.get_stats(prediction.long(), refLayer.long(), mode="binary")

    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro").item()

    return iou_score, f1_score, accuracy


def mean_median(IoU, F1, A):
    IoU_mean = statistics.mean(IoU)
    IoU_median = statistics.median(IoU)
    F1_mean = statistics.mean(F1)
    F1_median = statistics.median(F1)
    A_mean = statistics.mean(A)
    A_median = statistics.median(A)
    
    print('IoU (mean, median)', IoU_mean, IoU_median, '\n', 'F1 (mean, median)', F1_mean, F1_median, '\n', 'A (mean, median)', A_mean, A_median)


    


