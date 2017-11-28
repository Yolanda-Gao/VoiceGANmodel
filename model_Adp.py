## New Modified model of my own. Time: Oct. 8th 

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import ipdb

import numpy as np

import pdb

kernel_sizes = [4,3,3]
strides = [2,2,1]
paddings=[0,0,1]

latent_dim = 300

## Style Discriminator
class StyleDiscriminator(nn.Module):
    def __init__(
            self, num_gpu
            ):

        super(StyleDiscriminator, self).__init__()
        self.num_gpu = num_gpu

        self.main = nn.Sequential(
                    nn.Conv2d(1, 32, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(32, 64, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64 * 2),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64 * 4),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64 * 8),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.AdaptiveMaxPool2d(1),
                    nn.Conv2d(512,64,kernel_size=(1,1)),
                    nn.Conv2d(64,2,kernel_size=(1,1))

                    )

    def forward(self, input):
        return self.main( input )


## output one value
class Discriminator(nn.Module):
    def __init__(
            self, num_gpu
            ):

        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu        

        self.main = nn.Sequential(
                    nn.Conv2d(1, 32, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(32, 64, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64 * 2),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64 * 4),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64 * 8),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.AdaptiveMaxPool2d(1),
                    nn.Conv2d(512,64,kernel_size=(1,1)),
                    nn.Conv2d(64,8,kernel_size=(1,1)),
                    nn.Conv2d(8,1,kernel_size=(1,1)),
                    nn.Sigmoid()
                    )

    def forward(self, input):
        return self.main( input )
 
# Varied Input Generator
class Generator(nn.Module):
    def __init__(
            self,
            num_gpu,
            ):

        super(Generator, self).__init__()
        self.num_gpu = num_gpu
               
 	# Input 256X256 (DEFAULT) to 128x128
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

	# 128x128 to 64x64
        self.conv2 = nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64 * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

	# 64x64 to 32x32
        self.conv3 = nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64 * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

	# 32x32 to 16x16
        self.conv4 = nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64 * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

	# 16x16 to 8x8
        self.conv5 = nn.Conv2d(64 * 8, 64 * 8, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(64 * 8)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)


        # Varied length feature inside (8x8 to 4x4)
        self.conv6 = nn.Conv2d(64 * 8, 64 * 8, 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(64 * 8)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)

        # 4x4 to 8x8
        self.tconv6 = nn.ConvTranspose2d(64 * 8, 64 * 8, 4, 2, 1, bias=False) 
        self.tbn6 = nn.BatchNorm2d(64 * 8) 
	self.trelu6 = nn.ReLU(True) 
 
        # 8x8 to 16x16
	self.tconv5 = nn.ConvTranspose2d(64 * 8, 64 * 8, 4, 2, 1, bias=False) 
	self.tbn5 = nn.BatchNorm2d(64 * 8) 
	self.trelu5 = nn.ReLU(True) 

        # 16x16 to 32x32
	self.tconv4 = nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False) 
	self.tbn4 = nn.BatchNorm2d(64 * 4) 
	self.trelu4 = nn.ReLU(True) 

        # 32x32 to 64X64
	self.tconv3 = nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False) 
	self.tbn3 = nn.BatchNorm2d(64 * 2) 
	self.trelu3 = nn.ReLU(True) 

        # 64x64 to 128X128
	self.tconv2 = nn.ConvTranspose2d(64 * 2,     64, 4, 2, 1, bias=False) 
	self.tbn2 = nn.BatchNorm2d(64) 
	self.trelu2 = nn.ReLU(True) 

        # 128x128 to 256X256
	self.tconv1 = nn.ConvTranspose2d(    64,      1, 4, 2, 1, bias=False) 


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

        ## Transposed CNN
 
        tconv6 = self.tconv6(relu6)
        tbn6 = self.tbn6( tconv6 )
        trelu6 = self.trelu6(tbn6)

        tconv5 = self.tconv5(trelu6)
        tbn5 = self.tbn5(tconv5) 
        trelu5 = self.trelu5(tbn5) 

        tconv4 = self.tconv4(trelu5) 
        tbn4 = self.tbn4(tconv4)
        trelu4 = self.trelu4(tbn4) 

        tconv3 = self.tconv3(trelu4) 
        tbn3 = self.tbn3(tconv3) 
        trelu3 = self.trelu3(tbn3) 

        tconv2 = self.tconv2(trelu3)
        tbn2 = self.tbn2(tconv2)
        trelu2 = self.trelu2(tbn2)

        tconv1 = self.tconv1(trelu2)

        # pdb.set_trace()
        return torch.sigmoid( tconv1 ), [relu1, relu2, relu3, relu4, relu5], [trelu2, trelu3, trelu4, trelu5, trelu6] 
