import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import random
import torch.nn.functional as F
import math

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU(0.05) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.BatchNorm2d(gate_channels)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.LeakyReLU(0.05), 
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
    def forward(self, x):
        avg_values = self.avg_pool(x)
        out = self.mlp(avg_values)      

        scale = torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x*scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) 
        return x*scale



class MAnet_SA(nn.Module):
    def __init__(self, inchannels, outchannels, encorder='resnet50', meteorologicalNum = 7):
        super(MAnet_SA, self).__init__()
        self.MAnet = smp.MAnet(encoder_name=encorder, in_channels=inchannels, classes=outchannels)
        self.norm1d = nn.BatchNorm1d(meteorologicalNum)
        self.norm2d = nn.BatchNorm2d(13)  

        
        self.expand = nn.Sequential(
            nn.Conv2d(meteorologicalNum, meteorologicalNum, 3, 1, 1, bias=False),
            nn.BatchNorm2d(meteorologicalNum),
        )
        
        self.MLP = nn.Sequential(       
        nn.Linear(in_features=20, out_features=10),
        nn.ReLU(),   
        nn.Linear(in_features=10, out_features = 1),
        )

        self.ChannelGate = ChannelGate(meteorologicalNum, reduction_ratio=3)
        self.SpatialGate = SpatialGate()
        self.finalCov = nn.Conv2d(meteorologicalNum, meteorologicalNum, 3, 1, 1, bias=False)


    def forward(self, outsideInput, currentInput):   

        # adjust the shape of x and add non-linearity
        x = outsideInput
        x = self.norm1d(outsideInput)  # normalize the neighboring data matrix
        x = self.MLP(x)    # the shape after this will be Batch x 7 x 1
        
        # expanding the 3d outside tensor to match the 4d currentInput tensor
        d1, d2, _ = x.shape
        _, _, H, W = currentInput.shape 
        x = x.unsqueeze(-1).expand(d1, d2, H, W)  # the shape after this will be Batch x 7 x H x W
        
        # add convolution layer to add varialces among the 2D plane (done for all 7 planes all at once)
        x = self.expand(x)      
        

        y = currentInput
        y = self.norm2d(currentInput)  # normalize localized input


        # add attention to neighboring data layers
        x = self.ChannelGate(x)
        x= self.SpatialGate(x)
        x = self.finalCov(x)   
        
        # add neighboring layers to the corresponding channels of the localized layers
        y[:, 5:12, :, :] = y[:, 5:12, :, :].add(x)      
        

        # feed the sum to MAnet
        y = self.MAnet(y)      

        return y