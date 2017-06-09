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
        model = models.vgg11(pretrained=True)
        model.features._modules['2'] = nn.MaxPool2d((4, 4), stride=(4, 4), dilation=(1, 1))
        model.classifier._modules['0'] = nn.Linear(4608, 512)
        model.classifier._modules['6'] = nn.Linear(512, FEATURE_SIZE)
        del model.classifier._modules['2']
        del model.classifier._modules['3']
        del model.classifier._modules['4']
        self.RGBStream = deepcopy(model)
        self.FlowStream = deepcopy(model)
        self.FlowStream.features._modules['0'] = nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        resetModel(self.FlowStream)
        self.hidden_size = HIDDEN_SIZE
        self.lstm = torch.nn.LSTM(FEATURE_SIZE, self.hidden_size, 2, bidirectional=True)
        self.rgbdropout = nn.Dropout(0.5)
        self.flowdropout = nn.Dropout(0.5)
        self.pool = nn.AvgPool1d(8)

    def reset_hidden(self, batch=1):
        self.h = (Variable(torch.zeros(2*2, batch, self.hidden_size)).cuda(), Variable(torch.zeros(2*2, batch, self.hidden_size)).cuda())

    def forward(self, rgb, flow):
        # Since we are batching sequences we can reset each time
        self.reset_hidden()
        rgbout = self.RGBStream(rgb)
        rgbout = self.rgbdropout(rgbout)
        flowout = self.FlowStream(flow)
        flowout = self.flowdropout(flowout)
        #feat = flowout
        feat = rgbout
        #feat = torch.cat([rgbout, flowout], dim=1)
        feat = feat.unsqueeze(1)
        #feat = self.pool(feat)
        #print(feat.size(), self.h[0].size())
        out, self.h = self.lstm(feat, self.h)
        #print(out.size())
        return out.squeeze(1)

