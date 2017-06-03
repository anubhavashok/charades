import torch
from torch import nn
from copy import deepcopy
from torchvision import models
from utils import *
from config import *

class TwoStreamNetwork(nn.Module):
    def __init__(self):
        super(TwoStreamNetwork, self).__init__()
        model = models.resnet18(pretrained=True)
        model._modules['fc'] = nn.Linear(512, FEATURE_SIZE)
        model._modules['avgpool'] = nn.AvgPool2d(5, 5)
        self.RGBStream = deepcopy(model)
        self.FlowStream = deepcopy(model)
        self.FlowStream._modules['conv1'] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resetModel(self.FlowStream)

    def forward(self, rgb, flow):
        rgbout = self.RGBStream(rgb)
        flowout = self.FlowStream(flow)
        out = torch.cat([rgbout, flowout], dim=1)
        return out
