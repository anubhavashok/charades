import torch
from torch import nn
from torch.autograd import Variable
from copy import deepcopy
from torchvision import models
from utils import *
from config import *
from torch.utils.serialization import load_lua
import sys

#sys.path.insert(0, '../torch_models/')

from torch_models.lstm_flow_modified import lstm_flow_modified 
from torch_models.lstm_rgb_modified import lstm_rgb_modified

class TwoStreamNetworkLSTM(nn.Module):
    def __init__(self):
        super(TwoStreamNetworkLSTM, self).__init__()
        self.RGBStream = lstm_rgb_modified
        self.RGBStream.load_state_dict(torch.load('./torch_models/lstm_rgb_modified.pth'))
        self.FlowStream = lstm_flow_modified
        self.FlowStream.load_state_dict(torch.load('./torch_models/lstm_flow_modified.pth'))
        FEATURE_SIZE=4096
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

