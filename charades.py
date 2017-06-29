import torch
from torchvision import transforms
from torch import nn, optim
from torch.nn import MSELoss, KLDivLoss, NLLLoss, CrossEntropyLoss, SmoothL1Loss, MultiLabelSoftMarginLoss, MultiLabelMarginLoss
from dataset import CharadesLoader
from copy import deepcopy
from torch.autograd import Variable
import torch.nn.functional as F
from torchnet import meter

from config import *
from utils import *
import numpy as np
from time import time

import argparse
parser = argparse.ArgumentParser(description='Lets win charades')
parser.add_argument('-name', type=str, required=False, default="No name provided", help='Name of experiment')
parser.add_argument('-resume', type=str, required=False, default=None, help='Path to resume model')

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
actionClassifier = getActionClassifier() 
transformer = getTransformer() 
if USE_LSTM:
    #from models.vgg16_twostream_lstm import TwoStreamNetworkLSTM
    from models.rgb_vgg16_twostream_lstm import TwoStreamNetworkLSTM
    net = TwoStreamNetworkLSTM()
else:
    from models.vgg_twostream import TwoStreamNetwork
    net = TwoStreamNetwork()

resume_epoch = 0
if args.resume:
    model = torch.load(args.resume)
    net = model['net']
    actionClassifier = model['classifier']
    transformer = model['transformer']
    resume_epoch = model['epoch']

if USE_GPU:
    net = net.cuda()
    actionClassifier = actionClassifier.cuda()
    transformer = transformer.cuda()

parametersList = [{'params': transformer.parameters()},
                  {'params': net.parameters()},
                  {'params': actionClassifier.parameters()}]
optimizer = getOptimizer(parametersList) 

if CLIP_GRAD:
    for p in actionClassifier.parameters():
        clip_grad(p, -1, 1)

kwargs = {'num_workers': 1, 'pin_memory': True}

predictionLossFunction = getPredictionLossFn()
recognitionLossFunction = getRecognitionLossFn()

def train():
    global actionClassifier
    global net
    net.train()
    actionClassifier.train()
    cl = CharadesLoader(DATASET_PATH, split="train", frame_selection='SPACED')
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(classbalanceweights, len(cl))
    train_loader = torch.utils.data.DataLoader(cl, shuffle=True, **kwargs)
    for epoch in range(resume_epoch, EPOCHS):
        adjust_learning_rate(optimizer, epoch)
        start = time()
        print('Training for epoch %d' % (epoch))
        for batch_idx, (data, target) in enumerate(train_loader):
            if LAMBDA > 0:
                toggleOptimization(optimizer, batch_idx, toggleFreq=10)
            (rgb, flow) = data
            rgb = rgb.squeeze(0)
            flow = flow.squeeze(0)
            target = target[0].squeeze()
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
            predictionLoss = predictionLossFunction(predFeature, nextFeature)
            _, action = torch.max(actionFeature, 1)
            #actionFeature[(target == 157).data.cuda().repeat(1, 158)] = 0
            recognitionLoss = recognitionLossFunction(actionFeature, target)
            jointLoss = recognitionLoss + LAMBDA * predictionLoss
            jointLoss.backward()
            if batch_idx % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()
            print(batch_idx, float(jointLoss.data.cpu().numpy()[0]))
            if INTERMEDIATE_TEST and (batch_idx+1) % INTERMEDIATE_TEST == 0:
                print('Intermediate testing: ', test(intermediate=True))
        print('Time elapsed %f' % (time() - start))
        if epoch % TEST_FREQ == 0:
            print('Test epoch %d:' % epoch)
            #torch.save({'net': net,
            #            'classifier': actionClassifier
            #           }, 'checkpoints/checkpoint%d.net' % epoch)
            mean_ap, acc = test()
            if SAVE_MODEL:
                saveModel(net, actionClassifier, transformer, mean_ap, epoch)
            print('acc: ', acc)

def test(intermediate=False):
    #mtr = meter.ConfusionMeter(k=NUM_ACTIONS)
    #mapmtr = meter.mAPMeter()
    outputs = []
    targets = []
    global actionClassifier
    global net
    net.eval()
    actionClassifier.eval()
    corr = 0
    t5cum = 0
    f = open('results/%s'%(OUTPUT_NAME), "w+")
    val_loader = torch.utils.data.DataLoader(CharadesLoader(DATASET_PATH, split="val", frame_selection='TEST'))
    for batch_idx, (data, target) in enumerate(val_loader):
        print(batch_idx)
        if intermediate and batch_idx == 1700:
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
        _, target_m = torch.max(target, 1)
        _, action = torch.max(actionFeature, 1)
        output = actionFeature.data.cpu().numpy()
        output = np.exp(output)
        output = np.divide(output, np.expand_dims(output.sum(1), axis=1))
        outputs.append(np.mean(output, 0))
        targets.append(np.max(target.data.cpu().numpy(), 0))
        correct = target_m.eq(action.type_as(target_m)).sum().data.cpu().numpy()
        corr += (100. * correct) / curRGB.size(0)
    #np.savetxt('cmatrix.txt', mtr.value(), fmt="%.2e")
    #print(mapmtr.value())
    #print(mtr.value())
    #plot_confusion_matrix(mtr.value(), [])
    #print(corr/(batch_idx))
    outputs = np.array(outputs)
    targets = np.array(targets)
    ap = charades_ap(outputs, targets)
    mean_ap = np.mean(ap)
    print('mAP', mean_ap)
    #print('Top5: ', 100*t5cum/(batch_idx))
    f.close()
    net.train()
    actionClassifier.train()
    return mean_ap, (corr/(batch_idx), 100*t5cum/(batch_idx))

if __name__ == "__main__":
    # Print experiment details
    print_config()
    train()
