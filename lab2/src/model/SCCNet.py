# implement SCCNet model

import torch
import torch.nn as nn
import torch.nn.functional as F

# reference paper: https://ieeexplore.ieee.org/document/8716937
class SquareLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.square(x)

class LogLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.log(x)

class SCCNet(nn.Module):
    def __init__(self, numClasses=4, timeSample=500, Nu=22, C=22, Nc=22, Nt=1, dropoutRate=0.5):
        super(SCCNet, self).__init__()

        # First convolutional block: Spatial component analysis
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=Nu, kernel_size=(C, Nt))
        self.bn1 = nn.BatchNorm2d(Nu)
        
        # Second convolutional block: Spatio-temporal filtering
        self.conv2 = nn.Conv2d(Nu, out_channels=20, kernel_size=(1, 12), stride=1, padding=(0, 6))
        self.bn2 = nn.BatchNorm2d(20)
        
        # Square Layer
        self.square = SquareLayer()
        
        # Dropout Layer
        self.dropout1 = nn.Dropout(dropoutRate)
        self.dropout2 = nn.Dropout(dropoutRate)

        # Pooling layer: Temporal smoothing
        self.pool = nn.AvgPool2d(kernel_size=(1, 62), stride=(1, 12))
        
        self.log = LogLayer()
        
        # Fully connected layer
        self.fc = nn.Linear(self.get_size(C, timeSample, Nu, Nt), numClasses, bias=True)

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)

        x = self.square(x)
        x = self.dropout2(x)

        # Pooling layer
        x = self.pool(x)
        x = self.log(x)
        
        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.softmax(x, dim=1)

    # if needed, implement the get_size method for the in channel of fc layer
    def get_size(self, C, timeSample, Nu, Nt):
        # input size: (batch_size, 1, C, timeSample)
        channel = 1
        H = C
        W = timeSample
        
        # after conv1: (batch_size, Nu, C, timeSample - Nt + 1) = (300, 22, 1, 437)
        channel = Nu
        H = 1
        W = timeSample - Nt + 1
        
        # after conv2: (batch_size, 20, C, (437 - 12 + 2*6) / 1 + 1) = (300, 20, 1, 438)
        channel = 20
        H = 1
        W = (W - 12 + 2*6) // 1 + 1

        # after pooling: (batch_size, 20, 1, (438 - 62) / 12 + 1) = (batch_size, 20, 1, (438 - 62) / 12 + 1) = (300, 20, 1, 32)
        channel = 20
        H = 1
        W = (W - 62) // 12 + 1
        
        return channel * H * W