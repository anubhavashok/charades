import torch
from torchvision import transforms
from torch import nn, optim
from torch.nn import MSELoss, KLDivLoss, NLLLoss, CrossEntropyLoss, SmoothL1Loss
from dataset import CharadesLoader
from copy import deepcopy
from torch.autograd import Variable
import torch.nn.functional as F

from config import *
from utils import *

import argparse
parser = argparse.ArgumentParser(description='Lets win charades')
parser.add_argument('-name', type=str, required=False, default="No name provided", help='Name of experiment')

args = parser.parse_args()
print(args.name)

if USE_GPU:
    torch.cuda.set_device(TORCH_DEVICE)
cc = None
if LOG:
    from pycrayon import CrayonClient
    os.system('')
    cc = CrayonClient(hostname="server_machine_address")
net = None
actionClassifier = None
optimizer = None
if USE_LSTM:
    from models.twostream_lstm import TwoStreamNetworkLSTM
    net = TwoStreamNetworkLSTM()
    actionClassifier = nn.Linear(HIDDEN_SIZE*2, NUM_ACTIONS)
else:
    from models.twostream import TwoStreamNetwork
    net = TwoStreamNetwork()
    actionClassifier = nn.Linear(FEATURE_SIZE*2, NUM_ACTIONS)

if USE_GPU:
    net = net.cuda()
    actionClassifier = actionClassifier.cuda()

if OPTIMIZER == 'ADAM':
    optimizer = optim.Adam([
            {'params': net.parameters()}, 
            {'params': actionClassifier.parameters()}], lr=LR, weight_decay=5e-4)
elif OPTIMIZER == 'SGD':
    optimizer = optim.SGD([
            {'params': net.parameters()}, 
            {'params': actionClassifier.parameters()}], lr=LR, momentum=MOMENTUM, weight_decay=5e-4)
else:
    optimizer = optim.RMSprop([
            {'params': net.parameters()}, 
            {'params': actionClassifier.parameters()}], lr=LR, weight_decay=5e-4)

if CLIP_GRAD:
    for p in actionClassifier.parameters():
        clip_grad(p, -1, 1)

kwargs = {'num_workers': 1, 'pin_memory': True}

kldivLoss = KLDivLoss()
mseLoss = MSELoss()
nllLoss = NLLLoss()
ceLoss = CrossEntropyLoss()
smoothl1Loss = SmoothL1Loss()

def train():
    global actionClassifier
    global net
    net.train()
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
            elif PREDICTION_LOSS = 'SMOOTHL1':
                predictionLoss = smoothl1Loss(curFeature, nextFeature)
            else:
                predictionLoss = kldivLoss(F.softmax(curFeature),  F.softmax(nextFeature))
            _, action = torch.max(actionFeature, 1)
            recognitionLoss = ceLoss(actionFeature, target)
            jointLoss = recognitionLoss + LAMBDA * predictionLoss
            jointLoss.backward()
            optimizer.step()
            print(batch_idx, float(jointLoss.data.cpu().numpy()[0]))
            if INTERMEDIATE_TEST and (batch_idx+1) % INTERMEDIATE_TEST == 0:
                print('Intermediate testing: ', test(intermediate=True))
        if epoch % TEST_FREQ == 0:
            print('Test epoch %d:' % epoch)
            #torch.save({'net': net,
            #            'classifier': actionClassifier
            #           }, 'checkpoints/checkpoint%d.net' % epoch)
            test()

def top5acc(pred, target):
    pred = pred.cpu()
    target = target.cpu()
    _, i = torch.topk(pred, 5, dim=1)
    i = i.type_as(target)
    mn, _ = torch.max(i.eq(target.repeat(5, 1).t()), dim=1)
    acc = torch.mean(mn.float())
    return acc


def writeTestScore(f, vid, scores):
    # perform merging algorithm
    score = scores[0].data.clone().fill_(0)
    k = 0
    for i in range(len(scores)):
        _, j = torch.max(scores[i], 0)
        if j != 157:
            score += scores[i].data
            k += 1
    score /= k
    score = score.cpu().numpy().tolist()[:-1]
    f.write("%s %s\n" % (vid, ' '.join(map(str, score))))

def test(intermediate=False):
    global actionClassifier
    global net
    net.eval()
    corr = 0
    t5cum = 0
    f = open('results/testscores.txt', "w+")
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
        vid = val_loader.dataset.video_names[batch_idx]
        writeTestScore(f, vid, actionFeature)
        t5a = top5acc(actionFeature, target)
        t5cum += t5a
        _, action = torch.max(actionFeature, 1)
        correct = target.eq(action.type_as(target)).sum().data.cpu().numpy()
        corr += (100. * correct) / curRGB.size(0)
    print(corr/(batch_idx))
    print('Top5: ', 100*t5cum/(batch_idx))
    f.close()
    return (corr/(batch_idx), 100*t5cum/(batch_idx))

if __name__ == "__main__":
    # Print experiment details
    print_config()
    train()
