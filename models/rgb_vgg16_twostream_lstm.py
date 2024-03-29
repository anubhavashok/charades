import torch
from torch import nn
from torch.autograd import Variable
from copy import deepcopy
from torchvision import models
import config

class TwoStreamNetworkLSTM(nn.Module):
    def __init__(self):
        super(TwoStreamNetworkLSTM, self).__init__()
        print('Model: RGB/VGG16/LSTM')
        config.USE_FLOW=False
        config.FEATURE_SIZE = 4096
        model = torch.load('models/pretrained_rgb.net')#models.vgg16(pretrained=True)
        del model.classifier._modules['6']
        #model.classifier._modules['6']=nn.Linear(4096, FEATURE_SIZE)
        self.RGBStream = model
        self.hidden_size = config.HIDDEN_SIZE
        self.lstm = torch.nn.LSTM(config.FEATURE_SIZE, self.hidden_size, 2, bidirectional=True)
        self.rgbdropout = nn.Dropout(0.5)
        self.pool = nn.AvgPool1d(4)

    def reset_hidden(self, batch=1):
        self.h = (Variable(torch.zeros(2*2, batch, self.hidden_size)).cuda(), Variable(torch.zeros(2*2, batch, self.hidden_size)).cuda())

    def forward(self, rgb, flow, reset=True):
        if reset:
            # Since we are batching sequences we can reset each time
            self.reset_hidden()
        rgbout = self.RGBStream(rgb)
        #rgbout = self.rgbdropout(rgbout)
        feat = rgbout
        feat = feat.unsqueeze(1)
        #feat = self.pool(feat)
        out, self.h = self.lstm(feat, self.h)
        return out.squeeze(1)

