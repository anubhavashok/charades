import torch
from torchvision import transforms
from torch import nn, optim
from torch.nn import MSELoss, KLDivLoss, NLLLoss, CrossEntropyLoss
from dataset import CharadesLoader
from copy import deepcopy
from torch.autograd import Variable
import torch.nn.functional as F
from pycrayon import CrayonClient

from config import *
from utils import *

cc = None
if LOG:
    os.system('')
    cc = CrayonClient(hostname="server_machine_address")

if USE_LSTM:
    CLIP_GRAD=True
    from models.twostream_lstm import TwoStreamNetworkLSTM
    net = TwoStreamNetworkLSTM()
else:
    from models.twostream import TwoStreamNetwork
    net = TwoStreamNetwork()

actionClassifier = nn.Linear(FEATURE_SIZE*2, NUM_ACTIONS)
if USE_GPU:
    torch.cuda.device(TORCH_DEVICE)
    net = net.cuda()
    actionClassifier = actionClassifier.cuda()

if OPTIMIZER == 'ADAM':
    optimizer = optim.Adam([
            {'params': net.parameters()}, 
            {'params': actionClassifier.parameters()}], lr=LR)
else:
    optimizer = optim.SGD([
            {'params': net.parameters()}, 
            {'params': actionClassifier.parameters()}], lr=LR, momentum=MOMENTUM)

if CLIP_GRAD:
    for p in actionClassifier.parameters():
        clip_grad(p, -1, 1)

kwargs = {'num_workers': 1, 'pin_memory': True}


kldivLoss = KLDivLoss()
mseLoss = MSELoss()
nllLoss = NLLLoss()
ceLoss = CrossEntropyLoss()

def train():
    train_loader = torch.utils.data.DataLoader(CharadesLoader(DATASET_PATH, split="train"), shuffle=True, **kwargs)
    for epoch in range(EPOCHS):
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
            curFeature = net(rgb, flow)
            nextFeature = net(nextRGB, nextFlow).detach()
            actionFeature = actionClassifier(curFeature)
            if PREDICTION_LOSS == 'MSE':
                predictionLoss = mseLoss(curFeature, nextFeature)
            else:
                predictionLoss = kldivLoss(F.softmax(curFeature),  F.softmax(nextFeature))
            recognitionLoss = ceLoss(actionFeature, target)
            jointLoss = recognitionLoss + LAMBDA * predictionLoss
            jointLoss.backward()
            optimizer.step()
            print(batch_idx, float(jointLoss.data.cpu().numpy()[0])/rgb.size(0))
            if INTERMEDIATE_TEST and (batch_idx+1) % INTERMEDIATE_TEST == 0:
                print('Intermediate testing: ', test(intermediate=True))
        if epoch % TEST_FREQ == 0:
            print('Test epoch %d:' % epoch)
            test()

def test(intermediate=False):
    corr = 0
    val_loader = torch.utils.data.DataLoader(CharadesLoader(DATASET_PATH, split="val"))
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
        curFeature = net(curRGB, curFlow).detach()
        actionFeature = actionClassifier(curFeature).detach()
        _, action = torch.max(actionFeature, 1)
        correct = target.eq(action.type_as(target)).sum().data.cpu().numpy()
        corr += (100. * correct) / curRGB.size(0)
    print(corr/(batch_idx))

if __name__ == "__main__":
    # Print experiment details
    print_config()
    train()
