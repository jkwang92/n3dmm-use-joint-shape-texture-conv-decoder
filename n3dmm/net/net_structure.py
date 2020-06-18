import torch
from torch import nn
import numpy as np
import math
import os

class img_encoder(nn.Module):
    def __init__(self):
        super(img_encoder,self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3,32,3,2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,3,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128,3,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,3,2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,96,3,1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96,192,3,1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192,192,3,2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192,128,3,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,256,3,1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(6*6*256,256),
            nn.ReLU()
        )
        
    def forward(self,x):
    
        code = self.conv_net(x)
        code = torch.reshape(code,(1,6*6*256))
        code = self.fc_layer(code)
        
        return code

class pts_encoder(nn.Module):
    def __init__(self):
        super(pts_encoder,self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3,16,(1,1),1),
            nn.ReLU(),
            nn.Conv2d(16,16,(4,1),(4,1)),
            nn.ReLU(),
            nn.Conv2d(16,16,(1,1),1),
            nn.ReLU(),
            nn.Conv2d(16,16,(4,1),(4,1)),
            nn.ReLU(),
            nn.Conv2d(16,16,(1,1),1),
            nn.ReLU(),
            nn.Conv2d(16,16,(4,1),(4,1)),
            nn.ReLU(),
            nn.Conv2d(16,16,(1,1),1),
            nn.ReLU(),
            nn.Conv2d(16,16,(4,1),(4,1)),
            nn.ReLU(),
        )

#       pts_num: 53215——13303——3325——831——207
        
        self.fc_layer = self.fc_layer = nn.Sequential(
            nn.Linear(207*16,256),
            nn.ReLU()
        )
    
    def forward(self,x):
        
        code = self.conv_net(x)
        code = torch.reshape(code,(-1,207*16))
        code = self.fc_layer(code)

        return code

class pts_decoder(nn.Module):
    def __init__(self):
        super(pts_decoder,self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(256,207*16),
            nn.ReLU()
        )
        self.conv_net = nn.Sequential(
            nn.ConvTranspose2d(16,16,(4,1),(4,1),output_padding=(3,0)),
            nn.ReLU(),
            nn.Conv2d(16,16,(1,1),1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,16,(4,1),(4,1),output_padding=(1,0)),
            nn.ReLU(),
            nn.Conv2d(16,16,(1,1),1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,16,(4,1),(4,1),output_padding=(3,0)),
            nn.ReLU(),
            nn.Conv2d(16,16,(1,1),1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,16,(4,1),(4,1),output_padding=(3,0)),
            nn.ReLU(),
            nn.Conv2d(16,3,(1,1),1),
            nn.ReLU()
        )
    def forward(self,x):
    
        pts = self.fc_layer(x)
        pts = torch.reshape(pts,(1,16,207,1))
        pts = self.conv_net(pts)
    
        return pts
        
        
