import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# S0 only considers localized input
class MAnet_S0(nn.Module):
    def __init__(self, inchannels, outchannels, encorder='resnet50'):
        super(MAnet_S0, self).__init__()

        self.MAnet = smp.MAnet(encoder_name=encorder, in_channels=inchannels, classes=outchannels)
        self.norm2d = nn.BatchNorm2d(inchannels)
        

    def forward(self, currentInput):
        
        y = currentInput

        # normalize localized input
        y = self.norm2d(currentInput) 
        y = self.MAnet(y)     

        return y