import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SimpleLayerNet(nn.Module):
    def __init__(self):
        super(SimpleLayerNet, self).__init__()
        self.fc1 = nn.Linear(8, 20)  
        self.fc2 = nn.Linear(20, 5)  

    def forward(self, x):
        x = F.relu(self.fc1(x))  
        noise = torch.randn_like(x) * 3.0
        x = self.fc2(x+noise)
        return x


