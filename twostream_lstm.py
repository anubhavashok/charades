import torch
from torchvision import models
from torch import nn, optim
from torch.nn import MSELoss, KLDivLoss, NLLLoss, CrossEntropyLoss
from dataset import CharadesLoader
from copy import deepcopy
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import LSTM

NUM_ACTIONS = 157 + 1
FEATURE_SIZE = 2048#4096

LAMBDA = 0#0.3
epochs = 50

# TODO: Try lstm


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
        model.fc = nn.Sequential()
        model.avgpool = nn.AvgPool2d(5, 5)
        model.fc.add_module('embedding', nn.Linear(1024, FEATURE_SIZE))
        self.RGBStream = deepcopy(model)
        self.FlowStream = deepcopy(model)
        self.FlowStream.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resetModel(self.FlowStream)
        self.hidden_size = FEATURE_SIZE
        self.lstm = torch.nn.LSTM(FEATURE_SIZE*2, self.hidden_size, 2, bidirectional=True)
        
    def reset_hidden(self, batch=1):
        self.h = (Variable(torch.zeros(2*2, batch, self.hidden_size)), Variable(torch.zeros(2*2, batch, self.hidden_size)))
    
    def forward(self, rgb, flow):
        feat = torch.cat([self.RGBStream(rgb), self.FlowStream(flow)], dim=1)
        feat = feat.unsqueeze(1)
        #print(feat.size(), self.h[0].size())
        out, self.h = self.lstm(feat, self.h)
        #print(out.size())
        return out.squeeze(1)


twoStreamNetwork = TwoStreamNetwork()#.cuda()
actionClassifier = nn.Linear(FEATURE_SIZE*2, NUM_ACTIONS)#.cuda()
for p in actionClassifier.parameters():
    clip_grad(p, -1, 1)
optimizer = optim.Adam([{'params': twoStreamNetwork.parameters()}, {'params': actionClassifier.parameters()}], lr=0.001)

train_loader = CharadesLoader('.', split="train")
val_loader = CharadesLoader('.', split="val")

kldivLoss = KLDivLoss()
mseLoss = MSELoss()
nllLoss = CrossEntropyLoss()#NLLLoss()

totalLoss = 0

def trainStep(curRGB, nextRGB, curFlow, nextFlow, target):
    global totalLoss
    '''
        curRGB - RGB video frame at current timestep
        nextRGB - RGB video frame one second later
        curFlow - Optical flow frames around curRGB
        nextFlow - Optical flow frames around nextRGB
    '''
    optimizer.zero_grad()
    nextFeature = twoStreamNetwork(nextRGB, nextFlow).detach()
    curFeature = twoStreamNetwork(curRGB, curFlow)
    actionFeature = actionClassifier(curFeature)
    # Maybe use KL-divergence
    #predictionLoss = kldivLoss(F.softmax(curFeature),  F.softmax(nextFeature))
    predictionLoss = mseLoss(curFeature, nextFeature)
    target = Variable(torch.LongTensor([int(target)])).detach()#.cuda().detach()
    recognitionLoss = nllLoss(actionFeature, target)
    _, action = torch.max(actionFeature, 1)
    print(action.data.cpu().numpy()[0], target.data.cpu().numpy()[0])
    jointLoss = recognitionLoss + LAMBDA * predictionLoss
    jointLoss.backward()
    optimizer.step()
    totalLoss += jointLoss.data.cpu().numpy()[0]

def train():
    global totalLoss
    # Loop over dataset and obtain relevant inputs
    # TODO: modify this part
    for epoch in range(epochs):
        print('Training for epoch %d' % (epoch))
        for batch_idx, (data, target) in enumerate(train_loader):
            (rgb, flow) = data
            twoStreamNetwork.reset_hidden()
            nextRGB = Variable(rgb[1:, :, :])
            rgb = Variable(rgb[:-1, :, :])
            nextFlow = Variable(flow[1:, :, :])
            flow = Variable(flow[:-1, :, :])
            target = [int(t) for t in target][:-1]
            target = Variable(torch.LongTensor(target)).detach()
            optimizer.zero_grad()
            curFeature = twoStreamNetwork(rgb, flow)
            twoStreamNetwork.reset_hidden()
            nextFeature = twoStreamNetwork(nextRGB, nextFlow).detach()
            actionFeature = actionClassifier(curFeature)
            predictionLoss = mseLoss(curFeature, nextFeature)
            recognitionLoss = nllLoss(actionFeature, target)
            '''for i in range(1, rgb.size(0)):
                curRGB = Variable(rgb[i-1].unsqueeze(0))#.cuda()
                curFlow = Variable(flow[i-1].unsqueeze(0))#.cuda()
                nextRGB = Variable(rgb[i].unsqueeze(0)).detach()#.cuda()
                nextFlow = Variable(flow[i].unsqueeze(0)).detach()#.cuda()
                trainStep(curRGB, nextRGB, curFlow, nextFlow, target[i-1])
            '''
            jointLoss = recognitionLoss + LAMBDA * predictionLoss
            print(jointLoss)
            jointLoss.backward()
            optimizer.step()
            totalLoss += jointLoss.data.cpu().numpy()[0]
            #print(float(totalLoss)/rgb.size(0))
            totalLoss = 0
        if epoch % 1 == 0:
            torch.save(twoStreamNetwork, 'models/twoStream_epoch%d.net'%epoch)
            print('Test epoch %d:' % epoch)
            test()

def test():
    corr = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        #if batch_idx == 200:
        #    break
        (curRGB, curFlow) = data
        curRGB = Variable(curRGB)#.cuda()
        curFlow = Variable(curFlow)#.cuda()
        target = Variable(target)#.cuda()
        curFeature = twoStreamNetwork(curRGB, curFlow)
        actionFeature = actionClassifier(curFeature)
        _, action = torch.max(actionFeature, 1)
        correct = target.eq(action.type_as(target)).sum().data.cpu().numpy()
        corr += (100. * correct) / curRGB.size(0)
        #print((100. * correct) / curRGB.size(0))
    print(corr/(batch_idx))

if __name__ == "__main__":
    train()
