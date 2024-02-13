import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import rasterio
import csv
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

def getImg(fname):
    return rasterio.open(fname).read()


class CarvanaDataset(Dataset):  
    def __init__(self, f_dir, l, transform=None):
        self.f_dir = f_dir
        self.transform = transform
        self.images = []
        for im_fname in l:
            if not os.path.exists(os.path.join(f_dir, im_fname)):
                continue
            self.images.append(im_fname)                   
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.f_dir, self.images[index])
        imageO = getImg(img_path)

        # get locations where the pixel values are not valid on the target and previous SAR layers
        union0 = np.isnan(imageO[0])+np.isnan(imageO[-1])
        imageO[0][union0>0] = 0
        imageO[-1][union0>0] = 0

        SAR = imageO[0]
        SAR = np.expand_dims(SAR, axis=0)
        
        # correct some super large values of HAND with the largest reasonable HAND value (the largest HAND value less than 5000) 
        imageO[1][imageO[1]> 280.4805] = 280.4805        
        

        # compute the cumulative rainfall layers and fix potential value errors due to spatial interpolation
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
    

        
        if self.transform is not None:
            image = np.moveaxis(image, 0, -1)
            SAR = np.moveaxis(SAR, 0, -1)
            
            augmentations = self.transform(image=image, mask=SAR)
            image = augmentations["image"]
            sar = augmentations["mask"]
            
            image = np.moveaxis(image, -1, 0)
            sar = np.moveaxis(SAR, -1, 0)

            return torch.from_numpy(image.copy()), torch.from_numpy(sar.copy())
        else:
            return torch.from_numpy(image.copy()), torch.from_numpy(SAR.copy())


