import torch
import torch.nn as nn
import torchvision
from dataPrj_S0_data import CarvanaDataset
from torch.utils.data import DataLoader
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import numpy as np
import math
import seaborn as sns
from geomloss import SamplesLoss
from torch.autograd import Variable
import statistics
import json
from sklearn.metrics import mean_absolute_error as mae


checkpoint_dir = 'dataPrj_checkpoints/mergeInput'
filename = "S0_150"

def save_checkpoint(state, epoch):
    print("=> Saving checkpoint current epoch")
    checkpoint_path = checkpoint_dir+'/'+filename+'.pth.tar'
    torch.save(state, checkpoint_path)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
    

class NSE_Loss(torch.nn.Module):
    def __init__(self):
        super(NSE_Loss, self).__init__()

    def forward(self, output, target):
        targetMean = torch.nn.AdaptiveAvgPool2d(1)(target)
        loss = ((target-output)**2).sum()/(((target-targetMean)**2).sum())
        return loss

        
def get_loadersTrain(
    currentList,
    outsideList,
    batch_size,
    train_transform = None,
):
    
    train_ds = CarvanaDataset(currentList, outsideList,train_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
    )

    return train_loader


def get_loadersVali_Test(
    currentList,
    outsideList,
    batch_size,
):
    
    ds = CarvanaDataset(currentList, outsideList)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )

    return loader



def validation_loop(loader, model, loss_fn):

    model.eval()
    torch.cuda.empty_cache()
    count = 0
    loss = 0
    
    with torch.no_grad():
        for _, x_current, y in loader:
            x_current = x_current.cuda()
            outputs = model(x_current).cuda()
            target = y.cuda()
            
            outputs = outputs.detach().cpu()
            target = target.detach().cpu()            
            
            loss1 = loss_fn[0](outputs, target)
            loss3 = loss_fn[1](outputs, target)

            loss += (0.5*loss1.detach().item()+0.5*loss3.detach().item())
            count += 1

            getStastics(outputs, target)

    loss /= count

    print("Validation Loss:", loss)   

    return loss  

    
ApproxiEML = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

def save_predictions_as_imgs(
    loader, model, criterion, device, batch, fileName
):
    print('average loss: ', validation_loop(loader, model, criterion))
    model.eval()
    
    MAE, NSE, EML, AAI3 = [], [], [], []
    torch.cuda.empty_cache()
    with torch.no_grad(): 
        for idx, (_, x_current, y) in enumerate(loader):
            output = model(x_current.to(device))
            target = y.to(device)
            
            for itemidx in range(target.shape[0]):
                
                iMAE, iNSE= getStastics(output.to(device)[itemidx], target[itemidx])
                MAE.append(iMAE)
                NSE.append(iNSE)
                
                iEML = ApproxiEML(output[itemidx], target[itemidx]).item()
                EML.append(iEML)   

        MAE_arr = np.asarray(MAE)
        EML_arr = np.asarray(EML)
        NSE_arr = np.asarray(NSE)
        
        
        AAI3 = 1.0/(MAE_arr+1)+1.0/(EML_arr+1)+NSE_arr        
        

        mean_median(MAE, EML, NSE, AAI3)  



def getStastics(predLayer, refLayer):

    MAE = torch.abs(predLayer-refLayer).mean().item()
    RMSE = math.sqrt(((predLayer-refLayer)*(predLayer-refLayer)).mean().item())
    refMean = torch.nn.AdaptiveAvgPool2d(1)(refLayer)
    NSE = 1-(((refLayer-predLayer)**2).sum().item())/(((refLayer-refMean)**2).sum()).item()

    return MAE, NSE


def mean_median(MAE, EML, NSE, AAI3):
    MAE_mean = statistics.mean(MAE)
    MAE_median = statistics.median(MAE)
    EML_mean = statistics.mean(EML)
    EML_median = statistics.median(EML)
    NSE_mean = statistics.mean(NSE)
    NSE_median = statistics.median(NSE)
    AAI3_mean = statistics.mean(AAI3)
    AAI3_median = statistics.median(AAI3)
    
    print('MAE (mean, median)', MAE_mean, MAE_median, '\n', 'EML (mean, median)', EML_mean, EML_median, '\n', 'NSE (mean, median)', NSE_mean, NSE_median, '\n', 'AAI3 (mean, median)', AAI3_mean, AAI3_median)

    


        
