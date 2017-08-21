import torch
from torch import nn
from torch.autograd import Variable
from copy import deepcopy
import config

class TwoStreamNetworkLSTM(nn.Module):
    def __init__(self):
        super(TwoStreamNetworkLSTM, self).__init__()
        print('Model: RGB/INCEPTION/3D')
        config.USE_FLOW=False
        model = torch.load('models/I3D.net')
        self.RGBStream = model

    def forward(self, rgb, flow):
        rgb = rgb.permute(1, 0, 2, 3).unsqueeze(0)
        out = self.RGBStream(rgb)
        out = out.squeeze()
        out = out.unsqueeze(0)
        return out

