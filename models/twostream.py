import torch
from torch import nn
from copy import deepcopy
from torchvision import models
from utils import *
from config import *

class TwoStreamNetwork(nn.Module):
    def __init__(self):
        super(TwoStreamNetwork, self).__init__()
        print('Model: BOTH/ResNet18/CNN')
        model = models.resnet18(pretrained=True)
        model._modules['fc'] = nn.Linear(512, FEATURE_SIZE)
        model._modules['avgpool'] = nn.AvgPool2d(5, 5)
        model.layer1.add_module('2', nn.Dropout2d(0.3))
        model.layer2.add_module('2', nn.Dropout2d(0.3))
        model.layer3.add_module('2', nn.Dropout2d(0.3))
        model.layer4.add_module('2', nn.Dropout2d(0.3))
        self.RGBStream = deepcopy(model)
        self.FlowStream = deepcopy(model)
        self.FlowStream._modules['conv1'] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resetModel(self.FlowStream)
        #self.pool = nn.AvgPool1d(8)
        self.rgbdropout = nn.Dropout()
        self.flowdropout = nn.Dropout()

    def forward(self, rgb, flow):
        rgbout = self.RGBStream(rgb)
        #rgbout = self.rgbdropout(rgbout)
        flowout = self.FlowStream(flow)
        #flowout = self.flowdropout(flowout)
        out = torch.cat([rgbout, flowout], dim=1)
        #out = self.pool(out.unsqueeze(0)).squeeze(0)
        return out
