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
        print('Model: RGB/VGG16/LSTM')
        model = torch.load('models/pretrained_rgb.net')#models.vgg16(pretrained=True)
        model.classifier._modules['6']=nn.Linear(4096, FEATURE_SIZE)
        self.RGBStream = model
        self.hidden_size = HIDDEN_SIZE
        self.lstm = torch.nn.LSTM(FEATURE_SIZE, self.hidden_size, 2, bidirectional=True)
        self.rgbdropout = nn.Dropout(0.5)
        self.pool = nn.AvgPool1d(8)

    def reset_hidden(self, batch=1):
        self.h = (Variable(torch.zeros(2*2, batch, self.hidden_size)).cuda(), Variable(torch.zeros(2*2, batch, self.hidden_size)).cuda())

    def forward(self, rgb, flow):
        # Since we are batching sequences we can reset each time
        self.reset_hidden()
        rgbout = self.RGBStream(rgb)
        rgbout = self.rgbdropout(rgbout)
        feat = rgbout
        feat = feat.unsqueeze(1)
        #feat = self.pool(feat)
        out, self.h = self.lstm(feat, self.h)
        return out.squeeze(1)

