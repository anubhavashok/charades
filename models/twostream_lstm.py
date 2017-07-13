import torch
from torch import nn
from torch.autograd import Variable
from copy import deepcopy
from torchvision import models
from utils import *
from config import *

class TwoStreamNetworkLSTM(nn.Module):
    def __init__(self):
        super(TwoStreamNetworkLSTM, self).__init__()
        print('Model: TwoStream/ResNet18/LSTM')
        USE_RGB=True
        USE_FLOW=True
        model = models.resnet18(pretrained=True)
        model.fc = nn.Dropout(0)
        self.RGBStream = deepcopy(model)
        self.FlowStream = deepcopy(model)
        w = self.FlowStream.conv1.state_dict()
        w['weight'] = w['weight'].mean(dim=1).repeat(1,20,1,1)
        self.FlowStream.conv1 = nn.Conv2d(20, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.FlowStream.conv1.load_state_dict(w)
        self.hidden_size = HIDDEN_SIZE
        self.lstm = torch.nn.LSTM(512*2, self.hidden_size, 2, bidirectional=True)
        self.rgbdropout = nn.Dropout()
        self.flowdropout = nn.Dropout()

    def reset_hidden(self, batch=1):
        self.h = (Variable(torch.zeros(2*2, batch, self.hidden_size)).cuda(), Variable(torch.zeros(2*2, batch, self.hidden_size)).cuda())

    def forward(self, rgb, flow):
        # Since we are batching sequences we can reset each time
        self.reset_hidden()
        rgbout = self.RGBStream(rgb)
        rgbout = self.rgbdropout(rgbout)
        flowout = self.FlowStream(flow)
        flowout = self.flowdropout(flowout)
        feat = torch.cat([rgbout, flowout], dim=1)
        feat = feat.unsqueeze(1)
        #print(feat.size(), self.h[0].size())
        out, self.h = self.lstm(feat, self.h)
        #print(out.size())
        return out.squeeze(1)

