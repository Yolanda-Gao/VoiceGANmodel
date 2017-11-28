import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import ipdb

import numpy as np

kernel_sizes = [4,3,3]
strides = [2,2,1]
paddings=[0,0,1]

latent_dim = 300

class Discriminator(nn.Module):
    def __init__(
            self, num_gpu
            ):

        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu        

        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64 * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64 * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64 * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(64 * 8, 64 * 8, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(64 * 8)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)

        self.conv6 = nn.Conv2d(64 * 8, 64 * 8, 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(64 * 8)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)

        self.conv7 = nn.Conv2d(64 * 8, 2, 4, 1, 0, bias=False)

    def forward(self, input):
        conv1 = self.conv1( input )
        relu1 = self.relu1( conv1 )

        conv2 = self.conv2( relu1 )
        bn2 = self.bn2( conv2 )
        relu2 = self.relu2( bn2 )

        conv3 = self.conv3( relu2 )
        bn3 = self.bn3( conv3 )
        relu3 = self.relu3( bn3 )

        conv4 = self.conv4( relu3 )
        bn4 = self.bn4( conv4 )
        relu4 = self.relu4( bn4 )

        conv5 = self.conv5( relu4 )
        bn5 = self.bn5( conv5 )
        relu5 = self.relu5( bn5 )

        conv6 = self.conv6( relu5 )
        bn6 = self.bn6( conv6 )
        relu6 = self.relu6( bn6 )

        conv7 = self.conv7( relu6 )

        # return torch.sigmoid( conv7 )
        return conv7
