import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv, utils
from nnunet.network_architecture.neural_network import SegmentationNetwork


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=[[1,1,1],[1,1,1]], kernal_size=[[3,3,3],[3,3,3]],padding=None,dilation=1, dropout=False):
        super(BasicBlock, self).__init__()
        nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.padding_size=padding
        if self.padding_size is None:
            self.padding_size=[]
            for krnl in kernal_size:
                self.padding_size.append([1 if i == 3 else 0 for i in krnl])
        self.conv1 = nn.Conv3d(in_channel, out_channel,kernal_size[0],stride[0],self.padding_size[0],dilation)
        self.conv2 = nn.Conv3d(out_channel,out_channel,kernal_size[1],stride[1],self.padding_size[1],dilation)
        self.instnorm = nn.InstanceNorm3d(out_channel,**norm_op_kwargs)
        self.lrelu = nn.LeakyReLU(**nonlin_kwargs)
        if dropout == True:
            self.dropout = nn.Dropout3d(**dropout_op_kwargs)
        else:
            self.dropout = None

    def forward(self, x):
        # first conv step
        x = self.conv1(x)
        if not self.dropout is None:
            x = self.dropout(x)
        x = self.lrelu(self.instnorm(x))

        # second conv step
        x = self.conv2(x)
        if not self.dropout is None:
            x = self.dropout(x)

        return  self.lrelu(self.instnorm(x))

class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='trilinear', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size
        self.predict = True

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)

class Decoder(nn.Module):
    def __init__(self,x_channel,skip_channel,out_channel,scale_factor=None):
        super(Decoder,self).__init__()
        self.conv = BasicBlock(x_channel+skip_channel,out_channel)
        self.upsample = nn.ConvTranspose3d(x_channel,x_channel,kernel_size=scale_factor,stride=scale_factor,bias=False)
    
    def forward(self,x,x_skip):
        x = self.upsample(x)
        # print('after up:',x.shape)
        x = torch.cat((x,x_skip),dim=1)
        x = self.conv(x)
        return x

class skUNet(SegmentationNetwork):
    def __init__(self, in_channel=1, out_channel=2, deep_supervised=True,predict=False):
        super(skUNet, self).__init__()
        self.deep_supervised = deep_supervised
        self.predict=True
        self.do_ds = False
        self.conv_op = nn.Conv3d
        self.num_classes = 2

        self.encoder1 = BasicBlock(in_channel, 64, stride=[[1,1,1],[1,1,1]],kernal_size=[[1,3,3],[1,3,3]])  
        self.encoder2 = BasicBlock(64, 128, stride=[[1,2,2],[1,1,1]])  
        self.encoder3 = BasicBlock(128, 256, stride=[[2,2,2],[1,1,1]])

        self.bottleneck = BasicBlock(256,512,stride=[[2,2,2],[1,1,1]])
        
        self.decoder1 = Decoder(512,256,256,scale_factor=[2,2,2])
        self.decoder2 = Decoder(256,128,128,scale_factor=[2,2,2]) 
        self.decoder3 = Decoder(128,64,64,scale_factor=[1,2,2])
        
        self.seg_out = nn.Conv3d(64,out_channel,1, 1, 0, 1, 1,False)
        self.center_out = nn.Conv3d(64,out_channel,1, 1, 0, 1, 1,False)
        self.dm_out = nn.Conv3d(64,1,1, 1, 0, 1, 1,False)
            
    def get_device(self):
        if next(self.parameters()).device == "cpu":
            return "cpu"
        else:
            return next(self.parameters()).device.index

    def forward(self, x):
        seg_outputs=[]
        center_outputs=[]
        dm_outputs= []
        skips = []
        
        x = self.encoder1(x)
        skips.append(x)
        # print(x.shape)

        x = self.encoder2(x)
        skips.append(x)
        # print(x.shape)

        x = self.encoder3(x)
        skips.append(x)
        # print(x.shape)
    
        x = self.bottleneck(x)
        # print(x.shape)

        x = self.decoder1(x,skips[-1])
        x = self.decoder2(x,skips[-2])
        x = self.decoder3(x,skips[-3])

        seg_outputs.append(self.seg_out(x))  
        center_outputs.append(self.center_out(x))
        dm_outputs.append(self.dm_out(x))

        if self.predict:
            return seg_outputs[-1]
        return seg_outputs[-1],center_outputs[-1],dm_outputs[-1]


    