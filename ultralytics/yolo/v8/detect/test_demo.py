import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from ultralytics.yolo.utils.tal import dist2bbox, make_anchors

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Twopath_NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, g_channels, c2, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """

        super(_NonLocalBlockND, self).__init__()


        self.dimension = dimension
        self.sub_sample = sub_sample
        
        self.g_channels = g_channels
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.inter_g_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            self.inter_g_channels = g_channels//2
            
            if self.inter_channels == 0:
                self.inter_channels = 1
                
        self.g = nn.Conv2d(in_channels=self.g_channels, out_channels=self.inter_g_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_g_channels, out_channels=self.g_channels,
                        kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.g_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_g_channels, out_channels=self.g_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        """
        x[0]=nc
        x[1]=256
        """

        batch_size = x[0].size(0)
        x_1 = x[1]
        x = x[0]
        #print(f'x1: {x_1.size()} ')
        #print(f'xc: {x.size()} ')


        g_x = self.g(x_1).view(batch_size, self.inter_g_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        
        #print(f'after g_x: {g_x.size()}')

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        #print(f'after theta_x: {theta_x.size()}')
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        #print(f'after phi_x: {phi_x.size()}')
        f = torch.matmul(theta_x, phi_x)
        #print(f'after mul phi & theta : {f.size()}')
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        #print(f'after weight mul g: {y.size()}')
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_g_channels, *x_1.size()[2:])
        #print(f'back to (B, 1/2channels, H,W): {y.size()}')
        W_y = self.W(y)
        z = W_y + x_1
        
        return z
    
    
x_c = torch.rand(16,20,80,80)
x_1 = torch.rand(16,256,80,80)

NL = _NonLocalBlockND(20, 256, 256)

z = NL([x_c,x_1])

print(x_c)
print(x_1)
print(z)