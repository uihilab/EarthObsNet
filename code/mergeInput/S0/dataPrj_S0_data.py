import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import rasterio
import csv
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import json

def getImg(fname):
    return rasterio.open(fname).read()

def getOutsideInput(fname):
    outsideInput = []
    f = open(fname, 'r').read()
    data = json.loads(f)
    
    for key in data:
        outsideInput += data[key]
        

    out_np_arr = (np.array(outsideInput)).transpose()   
    
    out_np_arr[0][out_np_arr[0]<0] = 0
    precp2_1 = out_np_arr[1]-out_np_arr[0]
    precp3_2 = out_np_arr[2]-out_np_arr[1]
    precp5_3 = out_np_arr[3]-out_np_arr[2]
    precp7_5 = out_np_arr[4]-out_np_arr[3]
    
    
    precp2_1[precp2_1<0] = 0
    precp3_2[precp3_2<0] = 0
    precp5_3[precp5_3<0] = 0
    precp7_5[precp7_5<0] = 0
    
    out_np_arr[1] = precp2_1
    out_np_arr[2] = precp3_2
    out_np_arr[3] = precp5_3
    out_np_arr[4] = precp7_5
    
    return torch.tensor(out_np_arr.tolist())


class CarvanaDataset(Dataset):  
    def __init__(self, currentLocList, otherLocList, transform=None):
        self.images = []
        self.otherLoc = []
        self.transform = transform

        for i in range(len(currentLocList)):
            img = currentLocList[i]
            if os.path.exists(img):
                self.images.append(img)
            if (otherLocList is not None) and os.path.exists(otherLocList[i]):
                self.otherLoc.append(otherLocList[i])
        
    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, index):

        img_path = self.images[index]
        imageO = getImg(img_path)

        # removing nan values
        union0 = np.isnan(imageO[0])+np.isnan(imageO[-1])
        imageO[0][union0>0] = 0
        imageO[-1][union0>0] = 0


        SAR = imageO[0]
        SAR = np.expand_dims(SAR, axis=0)
        
        
        # correct nodata (represented by super large values) pixels of HAND
        imageO[1][imageO[1]> 280.4805] = 280.4805         
        
        imageO[3][imageO[3]<0] = 0
        interval2_1 = imageO[4]-imageO[3]
        interval3_2 = imageO[5]-imageO[4]
        interval5_3 = imageO[6]-imageO[5]
        interval7_5 = imageO[7]-imageO[6]
        
        interval2_1[interval2_1<0]=0
        interval3_2[interval3_2<0]=0
        interval5_3[interval5_3<0]=0
        interval7_5[interval7_5<0]=0
        
        
        image = imageO[[1,2,2,2,2,3,4,5,6,7,8,9,10], :, :]
        
        # conducting binary encoding for the categorical variable - landcover
        image[1] = 0  
        image[1][imageO[2] >= 80] = 1
        image[2] = 0 
        image[2][(imageO[2] >= 40)&(imageO[2] <= 70)] = 1
        image[3] = 0 
        image[3][(imageO[2]==20)|(imageO[2]==30)|(imageO[2]==60)|(imageO[2]==70)|(imageO[2]==95)|(imageO[2]==100)] = 1
        image[4] = 0 
        image[4][(imageO[2]==10)|(imageO[2]==30)|(imageO[2]==50)|(imageO[2]==70)|(imageO[2]==90)|(imageO[2]==100)] = 1


        image[6] = interval2_1
        image[7] = interval3_2
        image[8] = interval5_3  
        image[9] = interval7_5  


        # read in inputs from outside locations
        if self.otherLoc:
            input_outside = self.otherLoc[index]

            # switch to 'aggInput_avgavg' to use data generated with 2-step averaging.
            # 'aggInput' and 'aggInput_maxmax' corresponds to data generated with taking the sum and taking the maximum with the 2 steps, respectively.
            # We encourage readers to check our paper for more details regarding how the data from neighboring areas is generated. 
            input_outside = input_outside.replace('aggInput', 'aggInput_avgavg')

            outInput = getOutsideInput(input_outside)
        else:
            outInput = []
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=SAR)
            image = augmentations["image"]
            sar = augmentations["mask"]
            
            return outInput, torch.from_numpy(image.copy()), torch.from_numpy(sar.copy())
        else:
            return outInput, torch.from_numpy(image.copy()), torch.from_numpy(SAR.copy())

