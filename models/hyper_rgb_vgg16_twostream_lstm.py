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
        print('Model: HyperRGB/VGG16/LSTM')
        model = torch.load('models/pretrained_rgb.net')#models.vgg16(pretrained=True)
        model.classifier._modules['6']=nn.Linear(4096, FEATURE_SIZE)
        model.classifier._modules['0'] = nn.Linear(30528, 4096)
        self.outputs = []
        def hook(module, input, output):
            self.outputs.append(output)
        model.features[4].register_forward_hook(hook)
        model.features[16].register_forward_hook(hook)
        model.features[30].register_forward_hook(hook)
        self.RGBStream = model
        self.hidden_size = HIDDEN_SIZE
        self.lstm = torch.nn.LSTM(FEATURE_SIZE, self.hidden_size, 2, bidirectional=True)
        self.rgbdropout = nn.Dropout(0.5)
        self.pool = nn.AvgPool1d(4)
        self.pools = [nn.AvgPool2d(16), nn.AvgPool2d(8), nn.Dropout(0)]

    def reset_hidden(self, batch=1):
        self.h = (Variable(torch.zeros(2*2, batch, self.hidden_size)).cuda(), Variable(torch.zeros(2*2, batch, self.hidden_size)).cuda())

    def forward(self, rgb, flow):
        # Since we are batching sequences we can reset each time
        self.reset_hidden()
        rgbout = self.RGBStream.features(rgb)
        pooled = []
        batch = self.outputs[0].size(0)
        for i in range(len(self.outputs)):
            pooled.append(self.pools[i](self.outputs[i]).view(batch, -1))
        self.outputs = []
        rgbout = torch.cat(pooled, 1)
        rgbout = self.RGBStream.classifier(rgbout)
        rgbout = self.rgbdropout(rgbout)
        feat = rgbout
        feat = feat.unsqueeze(1)
        #feat = self.pool(feat)
        out, self.h = self.lstm(feat, self.h)
        return out.squeeze(1)

