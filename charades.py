import torch
from torchvision import transforms
from torch import nn, optim
from torch.nn import MSELoss, KLDivLoss, NLLLoss, CrossEntropyLoss, SmoothL1Loss, MultiLabelSoftMarginLoss
from dataset import CharadesLoader
from copy import deepcopy
from torch.autograd import Variable
import torch.nn.functional as F
from torchnet import meter

from config import *
from utils import *
import numpy as np

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
transformer = None
optimizer = None
if USE_LSTM:
    #from models.vgg16_twostream_lstm import TwoStreamNetworkLSTM
    from models.rgb_vgg16_twostream_lstm import TwoStreamNetworkLSTM
    net = TwoStreamNetworkLSTM()
    #actionClassifier = nn.Linear(HIDDEN_SIZE*2, NUM_ACTIONS)
    actionClassifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(HIDDEN_SIZE*2, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(HIDDEN_SIZE, NUM_ACTIONS)
    )
    transformer = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(HIDDEN_SIZE*2, HIDDEN_SIZE*2),
        nn.Dropout(0.5),
        nn.Linear(HIDDEN_SIZE*2, HIDDEN_SIZE*2)
    )
else:
    from models.vgg_twostream import TwoStreamNetwork
    net = TwoStreamNetwork()
    actionClassifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(FEATURE_SIZE*2, FEATURE_SIZE),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(FEATURE_SIZE, NUM_ACTIONS)
    )
    transformer = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(FEATURE_SIZE*2, FEATURE_SIZE*2),
        nn.Dropout(0.5),
        nn.Linear(FEATURE_SIZE*2, FEATURE_SIZE*2)
    )
    #actionClassifier = nn.Linear(FEATURE_SIZE*2, NUM_ACTIONS)

if USE_GPU:
    net = net.cuda()
    actionClassifier = actionClassifier.cuda()
    transformer = transformer.cuda()

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
nllLoss = NLLLoss(weight=invClassWeightstensor.cuda())
ceLoss = CrossEntropyLoss(weight=invClassWeightstensor.cuda())
mlsml = MultiLabelSoftMarginLoss()
smoothl1Loss = SmoothL1Loss()
tripletLoss = TripletLoss()


def train():
    global actionClassifier
    global net
    net.train()
    cl = CharadesLoader(DATASET_PATH, split="train", frame_selection='SPACED')
    train_loader = torch.utils.data.DataLoader(cl, shuffle=True, **kwargs)
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
            curFeature = net(rgb, flow)
            predFeature = transformer(curFeature)
            actionFeature = actionClassifier(curFeature)
            #print(np.argsort(actionFeature.data.cpu().numpy(), axis=1)[:, -5:])
            #print(np.argsort(target.data.cpu().numpy(), axis=1)[:, -5:])
            nextFeature = net(nextRGB, nextFlow).detach()
            if PREDICTION_LOSS == 'MSE':
                predictionLoss = mseLoss(predFeature, nextFeature)
            elif PREDICTION_LOSS == 'SMOOTHL1':
                predictionLoss = smoothl1Loss(predFeature, nextFeature)
            elif PREDICTION_LOSS == 'TRIPLET':
                negatives, _ = cl.randomSamples(curFeature.size(0))
                negativeFeature = net(Variable(negatives[0], requires_grad=False).cuda(), Variable(negatives[1], requires_grad=False).cuda()).detach()
                predictionLoss = tripletLoss(curFeature, nextFeature, negativeFeature)
            else:
                predictionLoss = kldivLoss(F.log_softmax(predFeature),  F.log_softmax(nextFeature))
            _, action = torch.max(actionFeature, 1)
            #actionFeature[(target == 157).data.cuda().repeat(1, 158)] = 0
            #recognitionLoss = nllLoss(F.log_softmax(actionFeature), target)
            #recognitionLoss = ceLoss(actionFeature, target)
            #print(F.log_softmax(curFeature), F.log_softmax(nextFeature))
            recognitionLoss = mlsml(actionFeature, target.float())
            jointLoss = recognitionLoss + LAMBDA * predictionLoss
            jointLoss.backward()
            if batch_idx % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()
            print(batch_idx, float(jointLoss.data.cpu().numpy()[0]))
            if INTERMEDIATE_TEST and (batch_idx+1) % INTERMEDIATE_TEST == 0:
                print('Intermediate testing: ', test(intermediate=True))
        if epoch % TEST_FREQ == 0:
            print('Test epoch %d:' % epoch)
            #torch.save({'net': net,
            #            'classifier': actionClassifier
            #           }, 'checkpoints/checkpoint%d.net' % epoch)
            test()

def test(intermediate=False):
    #mtr = meter.ConfusionMeter(k=NUM_ACTIONS)
    #mapmtr = meter.mAPMeter()
    outputs = []
    targets = []
    global actionClassifier
    global net
    net.eval()
    corr = 0
    t5cum = 0
    f = open('results/testscores.txt', "w+")
    val_loader = torch.utils.data.DataLoader(CharadesLoader(DATASET_PATH, split="val", frame_selection='SPACED'))
    for batch_idx, (data, target) in enumerate(val_loader):
        print(batch_idx)
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
        #mtr.add(actionFeature.data, target.data)
        #mapmtr.add(actionFeature.data, target.data.cpu().numpy())
        #mapmtr.add(actionFeature.data, target.data, target.data)
        #t5a = top5acc(actionFeature, target)
        t5a = 0
        t5cum += t5a
        _, target_m = torch.max(target, 1)
        _, action = torch.max(actionFeature, 1)
        outputs.append(np.mean(actionFeature.data.cpu().numpy(), 0))
        targets.append(np.max(target.data.cpu().numpy(), 0))
        correct = target_m.eq(action.type_as(target_m)).sum().data.cpu().numpy()
        corr += (100. * correct) / curRGB.size(0)
    #np.savetxt('cmatrix.txt', mtr.value(), fmt="%.2e")
    #print(mapmtr.value())
    #print(mtr.value())
    #plot_confusion_matrix(mtr.value(), [])
    #print(corr/(batch_idx))
    outputs = np.array(outputs)
    outputs = np.exp(outputs)
    outputs = np.divide(outputs, np.expand_dims(outputs.sum(1), axis=1))
    targets = np.array(targets)
    ap = charades_ap(outputs, targets)
    print('mAP', np.mean(ap))
    #print('Top5: ', 100*t5cum/(batch_idx))
    f.close()
    return (corr/(batch_idx), 100*t5cum/(batch_idx))

if __name__ == "__main__":
    # Print experiment details
    print_config()
    train()
