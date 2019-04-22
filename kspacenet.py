import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as f

class KspaceNet(nn.Module):
    def __init__(self,n_channels,n_outputs):
        super(KspaceNet, self).__init__()
        self.conv1 = nn.Conv2d(n_channels,32, 5)
        self.pool  = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 32, 5)
        self.conv3 = nn.Conv1d(32, 64, 5)
        self.fc1   = nn.Linear(59*128, 128)
        self.fc2   = nn.Linear(128, 32)
        self.fc3   = nn.Linear(32, n_outputs)
#         self.fc3   = nn.Linear(64, 7)
        self.sigmoid = nn.Sigmoid()
        
#         self.fc3   = nn.Linear(256, 1)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool(x1)
        x3 = self.relu(x2)
        x4 = self.conv2(x3)
        x5 = self.pool(x4)
        x6 = self.relu(x5)
        x7 = self.conv3(x6)
        x8 = self.pool(x7)
        x9 = self.relu(x8)
        x9 = x9.view(-1, 59*128)
        x10= self.fc1(x9)
        x11= self.relu(x10)
        x12= self.fc2(x11)
#         x13 = self.relu(x12)
        x13 = self.fc3(x12)
        x14 = f.normalize(x13, p=2, dim=1)
#         x13 = self.relu(x12)
#         x14 = self.fc3(x13)
#         x15 = self.sigmoid(x14)
#         norm1 = x12.norm(keepdim=True)
#         x13 = x12.div(norm1.expand_as(x12))
#         x13= self.relu(x12)
#         x14= self.fc3(x13)
        
        return x14