
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from math import sqrt


class Source(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, 2, padding = 2)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, 64, 5, 2, padding = 2)
        self.conv3= nn.Conv2d(64, 8, 5, 2, padding=2)
        self.fc1 = nn.Linear(864, 4)

        self.init_weights()

    
    def init_weights(self):
        # set seed
        torch.manual_seed(0)
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)
        
        C_in = self.fc1.weight.size(1)
        nn.init.normal_(self.fc1.weight, 0.0, 1/sqrt(C_in))
        nn.init.constant_(self.fc1.bias, 0.0)
        pass


    def forward(self, x):
        N, C, H, W = x.shape
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1) 
        z = self.fc1(x)

        return torch.Tensor(z)
        
class ProModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, 2, padding = 2)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, 64, 5, 2, padding = 2)
        self.conv3= nn.Conv2d(64, 8, 5, 2, padding=2)
        self.fc1 = nn.Linear(864, 4)
        self.softmax = nn.Softmax()

        self.init_weights()

    
    def init_weights(self):
        # set seed
        torch.manual_seed(0)
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)
        
        C_in = self.fc1.weight.size(1)
        nn.init.normal_(self.fc1.weight, 0.0, 1/sqrt(C_in))
        nn.init.constant_(self.fc1.bias, 0.0)
        pass


    def forward(self, x):
        N, C, H, W = x.shape
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1) 
        z = self.fc1(x)
        z = self.softmax(z)

        return torch.Tensor(z)
        