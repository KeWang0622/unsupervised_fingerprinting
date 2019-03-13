import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
class PixelNet(nn.Module):
    def __init__(self,n_channels):
        super(PixelNet, self).__init__()
        self.conv1 = nn.Conv1d(n_channels,8, 3)
        self.pool  = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.conv3 = nn.Conv1d(16, 32, 3)
        self.fc1   = nn.Linear(249*8*2, 512)
        self.fc2   = nn.Linear(512, 256)
#         self.fc3   = nn.Linear(256, 1)
        
    def forward(self, x):
        x1 = self.conv1(x)
#         x2 = self.pool(x1)
#         x3 = self.relu(x2)
#         x4 = self.conv2(x3)
#         x5 = self.pool(x4)
#         x6 = self.relu(x5)
#         x7 = self.conv3(x6)
#         x8 = self.pool(x7)
#         x9 = self.relu(x8)
        x9 = x1.view(-1, 249*8*2)
        x10= self.fc1(x9)
#         x11= self.relu(x10)
        x12= self.fc2(x10)
        norm1 = x12.norm(keepdim=True)
        x13 = x12.div(norm1.expand_as(x12))
#         x13= self.relu(x12)
#         x14= self.fc3(x13)
        
        return x13