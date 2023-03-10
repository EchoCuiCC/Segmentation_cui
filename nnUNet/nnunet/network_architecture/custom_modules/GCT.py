import torch
import torch.nn.functional as F
import math
from torch import nn


class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu
        self.num_channels = num_channels

    def forward(self, x):
        shape = x.shape
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2,3,4), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
            
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2,3,4), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')
            sys.exit()

        embedding = embedding * norm + self.beta
        gate = torch.softmax(embedding.view([2,shape[0],self.num_channels/2,1,1,1]),0)

        return gate