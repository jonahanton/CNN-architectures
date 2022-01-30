"""

Implementations of:

    i. ResNet18 - arXiv:1512.03385
    ii. ResNet18C/D - arXiv:1812.01187
    iii. SE-ResNet18 (Squeeze-Excitation) (+ SE-ResNet18C/D) - arXiv:1709.01507

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module): 
    def __init__(self, inchannel, outchannel, stride=1): 
        
        super(ResidualBlock, self).__init__() 
        
        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, 
                                         stride=stride, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel), 
                                  nn.ReLU(inplace=True), 
                                  nn.Conv2d(outchannel, outchannel, kernel_size=3, 
                                         stride=1, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel)) 
        
        self.shortcut = nn.Sequential() 
        
        if stride != 1 or inchannel != outchannel: 
            
            self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, 
                                                 kernel_size=1, stride=stride, 
                                                 padding = 0, bias=False), 
                                          nn.BatchNorm2d(outchannel) ) 
            
    def forward(self, x): 
        
        out = self.left(x) 
        out += self.shortcut(x) 
        out = F.relu(out) 
        
        return out


class ResidualBlockTweaked(nn.Module): 
    def __init__(self, inchannel, outchannel, stride=1): 
        
        super(ResidualBlock, self).__init__() 
        
        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, 
                                         stride=stride, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel), 
                                  nn.ReLU(inplace=True), 
                                  nn.Conv2d(outchannel, outchannel, kernel_size=3, 
                                         stride=1, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel)) 
        
        self.shortcut = nn.Sequential() 
        
        if stride != 1 or inchannel != outchannel: 
            
            self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),
                                          nn.Conv2d(inchannel, outchannel, 
                                                 kernel_size=1, stride=1, 
                                                 padding = 0, bias=False), 
                                          nn.BatchNorm2d(outchannel)) 
            
    def forward(self, x): 
        
        out = self.left(x) 
        out += self.shortcut(x) 
        out = F.relu(out) 
        
        return out


class ResNet18(nn.Module):
    
    def __init__(self, ResidualBlock, num_classes = 20):
        
        super(ResNet18, self).__init__()
        
        self.inchannel = 64
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size = 7, stride = 2,
                                            padding = 3, bias = False), 
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride = 1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride = 2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride = 2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
        
    
    def make_layer(self, block, channels, num_blocks, stride):
        
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
            
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet18C(nn.Module):
    
    def __init__(self, ResidualBlock, num_classes = 20):
        
        super(ResNet18C, self).__init__()
        
        self.inchannel = 64
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size = 3, stride = 2,
                                            padding = 1, bias = False), 
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride=1,
                                            padding=1, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True), 
                                nn.Conv2d(32, 64, kernel_size=3, stride=1,
                                            padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True))
                                 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride = 1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride = 2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride = 2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
        
    
    def make_layer(self, block, channels, num_blocks, stride):
        
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
            
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SELayer(nn.Module):

    def __init__(self, c, r):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEResidualBlock(nn.Module): 
    def __init__(self, inchannel, outchannel, stride=1, r=16): 
        
        super().__init__()
        
        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, 
                                         stride=stride, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel), 
                                  nn.ReLU(inplace=True), 
                                  nn.Conv2d(outchannel, outchannel, kernel_size=3, 
                                         stride=1, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel),
                                  SELayer(outchannel, r),
                                  ) 
        
        self.shortcut = nn.Sequential() 
        
        if stride != 1 or inchannel != outchannel: 
            
            self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, 
                                                 kernel_size=1, stride=stride, 
                                                 padding = 0, bias=False), 
                                          nn.BatchNorm2d(outchannel) ) 
            
    def forward(self, x): 
        
        out = self.left(x) 
        out += self.shortcut(x) 
        out = F.relu(out) 
        return out


class SEResidualBlockTweaked(nn.Module): 
    def __init__(self, inchannel, outchannel, stride=1, r=16): 
        
        super().__init__()
        
        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, 
                                         stride=stride, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel), 
                                  nn.ReLU(inplace=True), 
                                  nn.Conv2d(outchannel, outchannel, kernel_size=3, 
                                         stride=1, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel),
                                  SELayer(outchannel, r),
                                  ) 
        
        self.shortcut = nn.Sequential() 
        
        if stride != 1 or inchannel != outchannel: 
            
            self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),
                                          nn.Conv2d(inchannel, outchannel, 
                                                    kernel_size=1, stride=1, 
                                                    padding = 0, bias=False), 
                                          nn.BatchNorm2d(outchannel)) 
            
    def forward(self, x): 
        
        out = self.left(x) 
        out += self.shortcut(x) 
        out = F.relu(out) 
        return out


class SE_Resnet18(ResNet18):
    
    def __init__(self):
        super().__init__(ResidualBlock=SEResidualBlock)


class ResNet18D(ResNet18):

    def __init__(self):
        super().__init__(ResidualBlock=ResidualBlockTweaked)


class SE_ResNet18D(ResNet18):

    def __init__(self):
        super().__init__(ResidualBlock=SEResidualBlockTweaked)


if __name__ == "__main__":
    pass