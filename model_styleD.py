## Adp Model for D

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
  
                    # nn.Linear(1,64),
                    # nn.LeakyReLU(0.2, inplace=True),
                    # nn.Linear(64,2),
                    # nn.LeakyReLU(0.2, inplace=True),
                    # nn.Linear(32,8),
                    # nn.LeakyReLU(0.2, inplace=True),
                    # nn.Linear(8,2)   
                    # nn.softmax() 
                    )

    def forward(self, input):
        return self.main( input )

