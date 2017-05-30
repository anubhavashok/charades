import torch
from torchvision import models, transforms
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
        model.fc.add_module('embedding', nn.Linear(512, FEATURE_SIZE))
        self.RGBStream = deepcopy(model)
        self.FlowStream = deepcopy(model)
        self.FlowStream.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resetModel(self.FlowStream)
        
    def forward(self, rgb, flow):
        return torch.cat([self.RGBStream(rgb), self.FlowStream(flow)], dim=1)

lstm = torch.nn.LSTM(FEATURE_SIZE*2, FEATURE_SIZE*2, 2, bidirectional=True)

twoStreamNetwork = TwoStreamNetwork().cuda()
actionClassifier = nn.Linear(FEATURE_SIZE*2, NUM_ACTIONS).cuda()
optimizer = optim.Adam([{'params': twoStreamNetwork.parameters()}, {'params': actionClassifier.parameters()}], lr=0.001)
for p in actionClassifier.parameters():
    clip_grad(p, -1, 1)

train_loader = CharadesLoader('.', split="train", 
    input_transform = transforms.Compose([
        transforms.Scale(224),
    ]))
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
    curFeature = twoStreamNetwork(curRGB, curFlow)
    nextFeature = twoStreamNetwork(nextRGB, nextFlow).detach()
    actionFeature = actionClassifier(curFeature)
    # Maybe use KL-divergence
    #predictionLoss = kldivLoss(F.softmax(curFeature),  F.softmax(nextFeature))
    predictionLoss = mseLoss(curFeature, nextFeature)
    target = Variable(torch.LongTensor([int(target)]), volatile=True).cuda().detach()
    recognitionLoss = nllLoss(actionFeature, target)
    _, action = torch.max(actionFeature, 1)
    #print(action.data.cpu().numpy()[0], target.data.cpu().numpy()[0])
    jointLoss = recognitionLoss + LAMBDA * predictionLoss
    totalLoss += jointLoss.data.cpu().numpy()[0]
    jointLoss.backward()
    optimizer.step()

def train():
    global totalLoss
    # Loop over dataset and obtain relevant inputs
    # TODO: modify this part
    for epoch in range(epochs):
        print('Training for epoch %d' % (epoch))
        for batch_idx, (data, target) in enumerate(train_loader):
            (rgb, flow) = data
            for i in range(1, rgb.size(0)):
                curRGB = Variable(rgb[i-1].unsqueeze(0)).cuda()
                curFlow = Variable(flow[i-1].unsqueeze(0)).cuda()
                nextRGB = Variable(rgb[i].unsqueeze(0), volatile=True).cuda()
                nextFlow = Variable(flow[i].unsqueeze(0), volatile=True).cuda()
                trainStep(curRGB, nextRGB, curFlow, nextFlow, target[i-1])
            print(float(totalLoss)/rgb.size(0))
            totalLoss = 0
            if batch_idx % 5000 == 1:
                print('Intermediate testing: ')
                test(intermediate=True)
        if epoch % 1 == 0:
            #torch.save(twoStreamNetwork, 'models/twostream%d.net'%epoch)
            #torch.save(actionClassifier, 'models/actionClassifier%d.net'%epoch)
            #torch.save({'net': twoStreamNetwork, 'linear': actionClassifier}, 'models/epoch%d.net'%epoch)
            print('Test epoch %d:' % epoch)
            test(intermediate=True)

def test(intermediate=False):
    corr = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        if batch_idx == 200:
            break
        (curRGB, curFlow) = data
        curRGB = Variable(curRGB, volatile=True).cuda()
        curFlow = Variable(curFlow, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda()
        curFeature = twoStreamNetwork(curRGB, curFlow)
        actionFeature = actionClassifier(curFeature)
        _, action = torch.max(actionFeature, 1)
        correct = target.eq(action.type_as(target)).sum().data.cpu().numpy()
        corr += (100. * correct) / curRGB.size(0)
        #print((100. * correct) / curRGB.size(0))
    print(corr/(batch_idx))

if __name__ == "__main__":
    train()
