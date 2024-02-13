import torch
import torch.nn as nn
import torchvision
from dataset_dataProj import CarvanaDataset
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

checkpoint_dir = 'dataPrj_checkpoints'
filename = "DeepLabV3P"

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
    loader, model, criterion, device, batch, fileName='results'
):
    print('average loss: ', validation_loop(loader, model, criterion))
    model.eval()

    MAE, NSE, EML, AAI= [], [], [], []
    with torch.no_grad(): 

        for idx, (x, y) in enumerate(loader):
            output = model(x.to(device))
            target = y.to(device)
            
            for itemidx in range(target.shape[0]):

                # compute MAE, NSE, and EML (SD in the following papers: https://doi.org/10.1016/j.isprsjprs.2023.10.002
                # and https://doi.org/10.1016/j.isprsjprs.2023.11.021) for each sample.
                iMAE, iNSE= getStastics(output.to(device)[itemidx], target[itemidx])
                MAE.append(iMAE)
                NSE.append(iNSE)
                
                iEML = ApproxiEML(output[itemidx], target[itemidx]).item()
                EML.append(iEML)

                ## The four lines down below will save the synthesized images 
                ## to investigate the usefulness of predictions in downstream tasks (flood mapping in our study).
                ## Please check our paper for more details about experiment designing. 
                # img_id = idx * 64 + itemidx
                # with open(f'dataPrj_results/fig5/files/prediction_{img_id}.ASC', 'w') as outfile:
                #     out = output[itemidx].detach().cpu().numpy().flatten()
                #     outfile.write(' '.join(str(num) for num in out))  


        MAE_arr = np.asarray(MAE)
        EML_arr = np.asarray(EML)
        NSE_arr = np.asarray(NSE)
        
        
        AAI = 1.0/(MAE_arr+1)+1.0/(EML_arr+1)+NSE_arr       
    
        # save metrics
        with open(f"dataPrj_files/NSE_{fileName}.txt", "w") as NSE_file:
            json.dump(NSE, NSE_file)   
        with open(f"dataPrj_files/MAE_{fileName}.txt", "w") as MAE_file:
            json.dump(MAE, MAE_file)
        with open(f"dataPrj_files/EML_{fileName}.txt", "w") as EML_file:
            json.dump(EML, EML_file)
        with open(f"dataPrj_files/AAI3_{fileName}.txt", "w") as AAI_file:
            json.dump(AAI.tolist(), AAI_file)

        mean_median(MAE, EML, NSE, AAI)   





def getStastics(predLayer, refLayer):

    MAE = torch.abs(predLayer-refLayer).mean().item()

    refMean = torch.nn.AdaptiveAvgPool2d(1)(refLayer)
    NSE = 1-(((refLayer-predLayer)**2).sum().item())/(((refLayer-refMean)**2).sum()).item()

    return MAE, NSE


def mean_median(MAE, EML, NSE, AAI):
    MAE_mean = statistics.mean(MAE)
    MAE_median = statistics.median(MAE)
    EML_mean = statistics.mean(EML)
    EML_median = statistics.median(EML)
    NSE_mean = statistics.mean(NSE)
    NSE_median = statistics.median(NSE)
    AAI_mean = statistics.mean(AAI)
    AAI_median = statistics.median(AAI)
    
    print('MAE (mean, median)', MAE_mean, MAE_median, '\n', 'EML (mean, median)', EML_mean, EML_median, '\n', 'NSE (mean, median)', NSE_mean, NSE_median, '\n', 'AAI (mean, median)', AAI_mean, AAI_median)


    


