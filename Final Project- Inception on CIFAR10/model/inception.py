import os
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["InceptionModel", "InceptionModel"]


_InceptionModelOuputs = namedtuple(
    "InceptionModelOuputs", ["logits", "aux_logits2", "aux_logits1"]
)


def InceptionModel(pretrained=False, progress=True, device="cpu", aux_logits=False, **kwargs):

    model = InceptionModel(aux_logits=aux_logits)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/InceptionModel.pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model



class InceptionModel(nn.Module):

    def __init__(self, num_classes=10, aux_logits=False):
        super(InceptionModel, self).__init__()
        self.aux_logits = aux_logits


        self.conv1 = BasicConv2d(3, 192, kernel_size=3, stride=1, padding=1)

        self.inception1 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception2 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=False)

        self.inception3 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception5 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception6 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception7 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=False)

        self.inception8 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception9 = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)


    def forward(self, x):
   
        x = self.conv1(x)

        x = self.inception1(x)
        x = self.inception2(x)

        x = self.maxpool1(x)

        x = self.inception3(x)

        #First auxiliary branch.
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)

        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)

        #Second auxiliary branch.
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception7(x)

        x = self.maxpool2(x)

        x = self.inception8(x)
        x = self.inception9(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return _InceptionModelOuputs(x, aux2, aux1)
        return x


#This is the implementation of the Inception Module. Improved implementation is used rather than the naive one. 
#Every convolution has a 1x1 convolution beforehand to reduce channels, except the pooling branch.
#Pooling is performed first, then the 1x1 convolution.
#The outputs are concetenated at the end as different channels.
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


#This is the ending sequence of the Auxiliary Branches. 
#The outputs are treated as normal model outputs to calculate loss in module.py, expect only in training.
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (4, 4))

        x = self.conv(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x), inplace=True)

        x = F.dropout(x, 0.7, training=self.training)
        
        x = self.fc2(x)

        return x

