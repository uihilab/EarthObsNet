import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import rasterio
import csv
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

def getImg(fname, synthesized = False):
    if not synthesized:
        return rasterio.open(fname).read()
    else:
        values = []
        with open(os.path.join('dataPrj_results/fig5/files', fname), 'r') as file:
            for line in file:
                line_values = line.split(' ')
                values.extend([float(value) for value in line_values])
                
        return np.array(values).reshape((256, 256))        


class CarvanaDataset(Dataset):  
    def __init__(self, f_dir, l, transform=None):
        self.f_dir = f_dir
        self.transform = transform
        self.images = []
        self.slope = []
        self.labels = []
        for im_fname in l:
            if not os.path.exists(os.path.join(f_dir, im_fname)):
                continue
            self.images.append(im_fname) 
            slopePath = im_fname[:17] + 'slope' + im_fname[23:].replace('Img', 'slope')      
            labelPath = im_fname[:17] + 'labels' + im_fname[23:].replace('Img', 'label')

            self.slope.append(slopePath)
            self.labels.append(labelPath)     

        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.f_dir, self.images[index])
        slope_path = os.path.join(self.f_dir, self.slope[index])
        label_path = os.path.join(self.f_dir, self.labels[index])

        slope = getImg(slope_path)
        label = getImg(label_path)

        imageO = getImg(img_path)[[0, 1, 1], :, :]    # SAR + HAND + HAND
        
        #============= for synthesized SAR
#         newName = (self.images[index].split('/')[-1]).replace('Img', 'prediction')
#         newName = newName.replace('tif', 'ASC')

#         imageO[0] = getImg(newName, True)
        #=============

        imageO[2] = slope  # SAR + HAND + HAND -> SAR + HAND + Slope

        invalid = np.isnan(imageO[0])   # Get invalid pixels in the SAR layer

        for idx in range(3):
            imageO[idx][invalid>0] = 0  # Change the pixels values to 0 for all layers where the SAR pixels are invalid

        imageO[1][imageO[1]> 280.4805] = 280.4805  # Get rid of the extreme valus in the HAND layer. 
    
        
        if self.transform is not None:
            imageO = np.moveaxis(imageO, 0, -1)
            label = np.moveaxis(label, 0, -1)
            
            augmentations = self.transform(image=imageO, mask=label)
            image = augmentations["image"]
            floodMap = augmentations["mask"]
            
            image = np.moveaxis(image, -1, 0)
            floodMap = np.moveaxis(floodMap, -1, 0)

            return torch.from_numpy(image.copy()), torch.from_numpy(floodMap.copy())
        else:
            return torch.from_numpy(imageO.copy()), torch.from_numpy(label.copy())


