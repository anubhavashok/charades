import torch
from torch import nn
from torch.autograd import Variable
from copy import deepcopy
import config

class TwoStreamNetworkLSTM(nn.Module):
    def __init__(self):
        super(TwoStreamNetworkLSTM, self).__init__()
        print('Model: RGB/INCEPTION/LSTM')
        config.USE_FLOW=False
        model = torch.load('models/I3D.net')
        self.RGBStream = model

    def forward(self, rgb, flow, reset=True):
        rgbout = self.RGBStream(rgb)
        feat = rgbout
        return out.squeeze(1)

