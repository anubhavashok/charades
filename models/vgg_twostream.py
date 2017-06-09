import torch
from torch import nn
from copy import deepcopy
from torchvision import models
from utils import *
from config import *

class TwoStreamNetwork(nn.Module):
    def __init__(self):
        super(TwoStreamNetwork, self).__init__()
        model = models.vgg11(pretrained=True)
        model.features._modules['2'] = nn.MaxPool2d((4, 4), stride=(4, 4), dilation=(1, 1))
        model.classifier._modules['0'] = nn.Linear(4608, 512)
        model.classifier._modules['6'] = nn.Linear(512, FEATURE_SIZE)
        del model.classifier._modules['2']
        del model.classifier._modules['3']
        del model.classifier._modules['4']
        #del model.classifier._modules['5']
        #del model.classifier._modules['6']
        self.RGBStream = deepcopy(model)
        self.FlowStream = deepcopy(model)
        self.FlowStream.features._modules['0'] = nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
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
