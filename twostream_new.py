import torch
from torchvision import models, transforms
from torch import nn, optim
from torch.nn import MSELoss, KLDivLoss, NLLLoss, CrossEntropyLoss
from dataset import CharadesLoader
from copy import deepcopy
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import LSTM
import gc
import torch.backends.cudnn as cudnn
from gpustat_s import *
#torch.backends.cudnn.enabled = False
#cudnn.benchmark=False

NUM_ACTIONS = 157 + 1
FEATURE_SIZE = 256#4096

LAMBDA = 0.3
epochs = 50


def resetModel(m):
    if len(m._modules) == 0 and hasattr(m, 'reset_parameters'):
        m.reset_parameters()
        return
    for i in m._modules.values():
        resetModel(i)

def clip_grad(v, min, max):
    v.register_hook(lambda g: g.clamp(min, max))
    return v


class TwoStreamNetwork(nn.Module):
    def __init__(self):
        super(TwoStreamNetwork, self).__init__()
        model = models.resnet18(pretrained=True)
        model._modules['fc'] = nn.Linear(512, FEATURE_SIZE)
        #model.fc = nn.Sequential()
        model._modules['avgpool'] = nn.AvgPool2d(5, 5)
        #model.fc.add_module('embedding', nn.Linear(512, FEATURE_SIZE))
        self.RGBStream = deepcopy(model)
        self.FlowStream = deepcopy(model)
        self.FlowStream._modules['conv1'] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resetModel(self.FlowStream)
        
    def forward(self, rgb, flow):
        rgbout = self.RGBStream(rgb)
        flowout = self.FlowStream(flow)
        out = torch.cat([rgbout, flowout], dim=1)
        return out

twoStreamNetwork = TwoStreamNetwork().cuda()
actionClassifier = nn.Linear(FEATURE_SIZE*2, NUM_ACTIONS).cuda()
#optimizer = optim.Adam([
#        {'params': twoStreamNetwork.parameters()}, 
#        {'params': actionClassifier.parameters()}], lr=0.001)

optimizer = optim.SGD([
        {'params': twoStreamNetwork.parameters()}, 
        {'params': actionClassifier.parameters()}], lr=0.001, momentum=0.9)

#for p in actionClassifier.parameters():
#    clip_grad(p, -1, 1)
kwargs = {'num_workers': 1, 'pin_memory': True}


kldivLoss = KLDivLoss()
mseLoss = MSELoss()
nllLoss = NLLLoss()
ceLoss = CrossEntropyLoss()

totalLoss = 0

def train():
    train_loader = torch.utils.data.DataLoader(CharadesLoader('.', split="train"), shuffle=True, **kwargs)
    #global totalLoss
    for epoch in range(epochs):
        print('Training for epoch %d' % (epoch))
        for batch_idx, (data, target) in enumerate(train_loader):
            (rgb, flow) = data
            rgb = rgb.squeeze(0)
            flow = flow.squeeze(0)
            target = target[0]
            if rgb.size(0) <= 1:
                continue
            nextRGB = rgb[1:, :, :]
            nextRGB = Variable(nextRGB, requires_grad=False).cuda()
            rgb = rgb[:-1, :, :]
            rgb = Variable(rgb).cuda()
            nextFlow = flow[1:, :, :]
            nextFlow = Variable(nextFlow, requires_grad=False).cuda()
            flow = flow[:-1, :, :]
            flow = Variable(flow).cuda()
            target = Variable(target[:-1].long(), requires_grad=False).cuda().detach()
            optimizer.zero_grad()
            #show_memusage(device=0)
            curFeature = twoStreamNetwork(rgb, flow)
            nextFeature = twoStreamNetwork(nextRGB, nextFlow).detach()
            actionFeature = actionClassifier(curFeature)
            #predictionLoss = kldivLoss(F.softmax(curFeature),  F.softmax(nextFeature))
            predictionLoss = mseLoss(curFeature, nextFeature)
            recognitionLoss = ceLoss(actionFeature, target)
            #_, action = torch.max(actionFeature, 1)
            #print(action.data.cpu().numpy()[0], target.data.cpu().numpy()[0])
            jointLoss = recognitionLoss + LAMBDA * predictionLoss
            jointLoss.backward()
            optimizer.step()
            #show_memusage(device=0)
            print(batch_idx, float(jointLoss.data.cpu().numpy()[0])/rgb.size(0))
            #del data, target, curFeature, actionFeature, rgb, nextRGB, flow, nextFlow, jointLoss 
            if (batch_idx+1) % 1000 == 0:
                print('Intermediate testing: ')
                test(intermediate=True)
        if epoch % 1 == 0:
            print('Test epoch %d:' % epoch)
            test()

def test(intermediate=False):
    corr = 0
    val_loader = torch.utils.data.DataLoader(CharadesLoader('.', split="val"))
    for batch_idx, (data, target) in enumerate(val_loader):
        if intermediate and batch_idx == 200:
            break
        (curRGB, curFlow) = data
        curRGB = curRGB.squeeze(0)
        curFlow = curFlow.squeeze(0)
        target = target[0]
        curRGB = Variable(curRGB, volatile=True).cuda()
        curFlow = Variable(curFlow, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda()
        curFeature = twoStreamNetwork(curRGB, curFlow).detach()
        actionFeature = actionClassifier(curFeature).detach()
        _, action = torch.max(actionFeature, 1)
        correct = target.eq(action.type_as(target)).sum().data.cpu().numpy()
        corr += (100. * correct) / curRGB.size(0)
        #print((100. * correct) / curRGB.size(0))
    print(corr/(batch_idx))

if __name__ == "__main__":
    train()
