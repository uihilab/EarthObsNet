import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import random
import torch.nn.functional as F
import math
import admin_torch


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
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

    
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

# the transformer code was adapted from the transformer implementation by Aladdin Persson 
# that is availabile at https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/transformer_from_scratch     
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):

        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values) 
        keys = self.keys(keys) 
        queries = self.queries(query)  

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)


        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20")).cuda()

        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)


        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)
        

    def forward(self, value, key, query, mask):

        attention = self.attention(value, key, query, mask)

        x = self.dropout(attention + query)
        forward = self.feed_forward(x)
        out = self.dropout(forward + x)
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        meteorological,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Linear(meteorological, embed_size)

        self.embedding = PositionalEncoding(
            embed_size, 
            dropout,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(20*embed_size, meteorological)


    def forward(self, x, mask):
        N, seq_length = x.shape[0], x.shape[1]

        out = self.embedding(self.word_embedding(x))

        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        out = out.view(out.shape[0], 1, -1)
        out = self.fc_out(out)
        
        return out



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)    
    
class Transformer(nn.Module):
    def __init__(
        self,
        meteorological,
        embed_size=64,
        num_layers=1,
        forward_expansion=4,
        heads=8,
        dropout=0.1,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            meteorological,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
        )

    # will use encorder-only structure, as it fits our needs better.
    # check our paper for explanations regarding model designing (https://doi.org/10.1016/j.isprsjprs.2023.11.021).
    def forward(self, src):
        enc_src = self.encoder(src, mask=None)
        return enc_src


class MAnet_SS(nn.Module):
    def __init__(self, inchannels, outchannels, encorder='resnet50', meteorologicalNum = 7):
        super(MAnet_SS, self).__init__()
        self.MAnet = smp.MAnet(encoder_name=encorder, in_channels=inchannels, classes=outchannels)
        self.norm1d = nn.BatchNorm1d(meteorologicalNum)
        self.norm2d = nn.BatchNorm2d(inchannels)
        self.expandNorm2d = nn.BatchNorm2d(meteorologicalNum)
        self.meteo = meteorologicalNum
        self.norm2d2 = nn.BatchNorm2d(inchannels)

        self.expand = nn.Sequential(
            nn.Conv2d(meteorologicalNum, meteorologicalNum, 3, 1, 1, bias=False),
            nn.BatchNorm2d(meteorologicalNum),
        )

        self.ChannelGate = ChannelGate(meteorologicalNum, reduction_ratio=3)
        self.SpatialGate = SpatialGate()
        self.finalCov = nn.Conv2d(meteorologicalNum, meteorologicalNum, 3, 1, 1, bias=False)

        self.transformer = Transformer(meteorologicalNum)
        


    def forward(self, outsideInput, currentInput):

        # adjust the shape of x and add non-linearity
        x = outsideInput
        x = self.norm1d(outsideInput)      
        x = torch.permute(x, (0, 2, 1))
        pred = self.transformer(x)   # the shape after this should be Batch x 1 x 7   
        pred = torch.permute(pred, (0, 2, 1))   # the shape after this should be Batch x 7 x 1  

        # expanding the 3D neighboring input matrix to match the 4D localized input
        d1, d2, _ = pred.shape
        _, _, H, W = currentInput.shape 
        pred = pred.unsqueeze(-1).expand(d1, d2, H, W)   # the shape after this will be Batch x 7 x H x W

        # add convolution layer to add varialces among the 2D plane (done for all 7 planes all at once)
        pred = self.expand(pred)
        
        y = currentInput
        y = self.norm2d(currentInput)  # normalize localized input


        # add attention to neighboring data layers
        pred = self.ChannelGate(pred)
        pred= self.SpatialGate(pred)
        pred = self.finalCov(pred)

        # add neighboring layers to the corresponding channels of the localized layers
        # and feed the sum to MAnet
        y[:, 5:12, :, :] = y[:, 5:12, :, :].add(pred)        
        y = self.MAnet(y)      

        return y