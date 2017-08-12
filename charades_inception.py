import torch
from torchvision import transforms
from torch import nn, optim
from torch.nn import MSELoss, KLDivLoss, NLLLoss, CrossEntropyLoss, SmoothL1Loss, MultiLabelSoftMarginLoss, MultiLabelMarginLoss
from copy import deepcopy
from torch.autograd import Variable
import torch.nn.functional as F
from torchnet import meter

import numpy as np
from time import time
import config

torch.set_num_threads(4)

import argparse
parser = argparse.ArgumentParser(description='Lets win charades')
parser.add_argument('-name', type=str, required=False, default="No name provided", help='Name of experiment')
parser.add_argument('-resume', type=str, required=False, default=None, help='Path to resume model')

args = parser.parse_args()
print(args.name)

if config.USE_GPU:
    torch.cuda.set_device(config.TORCH_DEVICE)
cc = None
if config.LOG:
    from pycrayon import CrayonClient
    os.system('')
    cc = CrayonClient(hostname="server_machine_address")
net = None
if config.USE_LSTM:
    #from models.vgg16_twostream_lstm import TwoStreamNetworkLSTM
    #from models.twostream_lstm import TwoStreamNetworkLSTM
    #from models.flow_vgg16_twostream_lstm import TwoStreamNetworkLSTM
    from models.inflated_inception_twostream import TwoStreamNetworkLSTM
    #from models.global_average_rgb_vgg16_twostream_lstm import TwoStreamNetworkLSTM
    net = TwoStreamNetworkLSTM()
else:
    from models.vgg_twostream import TwoStreamNetwork
    net = TwoStreamNetwork()

from config import *
from utils import *

actionClassifier = getActionClassifier() 
transformer = getTransformer() 

resume_epoch = 0
if args.resume:
    model = torch.load(args.resume)
    net = model['net']
    actionClassifier = model['classifier']
    transformer = model['transformer']
    resume_epoch = model['epoch']

if USE_GPU:
    net = net.cuda()
    #net = nn.DataParallel(net, device_ids=[0, 1])
    actionClassifier = actionClassifier.cuda()
    #actionClassifier = nn.DataParallel(actionClassifier, device_ids=[0, 1])
    transformer = transformer.cuda()
    #transformer = nn.DataParallel(transformer, device_ids=[0, 1])

parametersList = [{'params': transformer.parameters()},
                  {'params': net.parameters()},
                  {'params': actionClassifier.parameters()}]
optimizer = getOptimizer(parametersList) 

if CLIP_GRAD:
    global clip_grad
    for params in parametersList:
        for p in params['params']:
            clip_grad(p, -1, 1)

kwargs = {'num_workers': 1, 'pin_memory': True}

from dataset_inception import InceptionDataset 
cl = InceptionDataset(DATASET_PATH, split="train")
predictionLossFunction = getPredictionLossFn(cl, net)
recognitionLossFunction = getRecognitionLossFn()

def train():
    global actionClassifier
    global net
    global transformer
    net.train()
    actionClassifier.train()
    transformer.train()
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(classbalanceweights, len(cl))
    train_loader = torch.utils.data.DataLoader(cl, shuffle=True, **kwargs)
    meter_rec = meter.AverageValueMeter()
    meter_pred = meter.AverageValueMeter()
    meter_joint = meter.AverageValueMeter()
    for epoch in range(resume_epoch, EPOCHS):
        meter_rec.reset()
        meter_pred.reset()
        meter_joint.reset()
        adjust_learning_rate(optimizer, epoch)
        start = time()
        print('Training for epoch %d' % (epoch))
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 10000:
                break
            (rgb, flow) = data
            rgb = rgb.squeeze(0)
            flow = flow.squeeze(0)
            target = target[0].squeeze()
            if rgb.size(0) <= 1:
                continue
            if LAMBDA > 0:
                toggleOptimization(optimizer, batch_idx, toggleFreq=16)
                nextRGB = rgb[1:, :, :]
                nextRGB = Variable(nextRGB, requires_grad=False).cuda()
                nextFlow = flow[1:, :, :]
                nextFlow = Variable(nextFlow, requires_grad=False).cuda()
            rgb = Variable(rgb).cuda()
            flow = Variable(flow).cuda()
            target = Variable(target.long(), requires_grad=False).cuda().detach()
            actionFeature = net(rgb, flow)
            #actionFeature = actionClassifier(curFeature)
            recognitionLoss = recognitionLossFunction(actionFeature, target)
            jointLoss = recognitionLoss
            if LAMBDA > 0:
                predFeature = transformer(curFeature)
                #af = Variable(torch.from_numpy(one_hot((target.size(0), NUM_ACTIONS), target.data)).float()).detach().cuda()
                #predFeature = transformer(torch.cat([af, curFeature], 1))
                #predFeature = transformer(torch.cat[softmax(actionFeature), curFeature], 1))
                #print(np.argsort(actionFeature.data.cpu().numpy(), axis=1)[:, -5:])
                #print(np.argsort(target.data.cpu().numpy(), axis=1)[:, -5:])
                nextFeature = net(nextRGB, nextFlow).detach()
                predictionLoss = predictionLossFunction(predFeature, nextFeature)
                #actionFeature[(target == 157).data.cuda().repeat(1, 158)] = 0
                jointLoss = recognitionLoss + LAMBDA * predictionLoss
                meter_pred.add(predictionLoss.data.cpu().numpy()[0])
            jointLoss.backward()
            meter_rec.add(recognitionLoss.data.cpu().numpy()[0])
            meter_joint.add(jointLoss.data.cpu().numpy()[0])
            _, action = torch.max(actionFeature, 1)
            if batch_idx % 250 == 0:
                print('%.2f%% [%d/%d] Recognition loss: %f, Prediction loss: %f, Joint loss: %f' % ((100. * batch_idx)/len(train_loader), batch_idx, len(train_loader), meter_rec.value()[0], meter_pred.value()[0], meter_joint.value()[0]))
                meter_rec.reset()
                meter_pred.reset()
                meter_joint.reset()
            if batch_idx % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()
            #print(batch_idx, float(jointLoss.data.cpu().numpy()[0]))
            if INTERMEDIATE_TEST and (batch_idx+1) % INTERMEDIATE_TEST == 0:
                print('Intermediate testing: ', test(intermediate=True))
        print('Time elapsed %f' % (time() - start))
        if epoch % TEST_FREQ == 0:
            print('Test epoch %d:' % epoch)
            mean_ap, acc = test()
            if SAVE_MODEL:
                saveModel(net, actionClassifier, transformer, mean_ap, epoch)
            print('acc: ', acc)

def test(intermediate=False):
    #mtr = meter.ConfusionMeter(k=NUM_ACTIONS)
    #mapmtr = meter.mAPMeter()
    scores = {}
    target_scores = {}
    outputs = []
    targets = []
    global actionClassifier
    global net
    net.eval()
    actionClassifier.eval()
    corr = 0
    t5cum = 0
    f = open('results/%s'%(OUTPUT_NAME), "w+")
    floc = open('results/loc_%s'%(OUTPUT_NAME), "w+")
    val_loader = torch.utils.data.DataLoader(InceptionDataset(DATASET_PATH, split="val"))
    print(len(val_loader))
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
        actionFeature = net(curRGB, curFlow).detach()
        vid = val_loader.dataset.snippets[batch_idx][0]
        # aggregate batches into map[vid]
        #actionFeature = actionClassifier(curFeature).detach()
        #actionFeature.data = unmapClasses(actionFeature.data)
        #vid = val_loader.dataset.video_names[batch_idx]
        #writeTestScore(f, vid, actionFeature)
        #writeLocScore(floc, vid, actionFeature)
        #mtr.add(actionFeature.data, target.data)
        #mapmtr.add(actionFeature.data, target.data.cpu().numpy())
        #mapmtr.add(actionFeature.data, target.data, target.data)
        #t5a = top5acc(actionFeature, target)
        #_, target_m = torch.max(target, 1)
        action, _ = torch.max(actionFeature, 0)
        output = actionFeature.data.cpu().numpy()
        # Remove softmax for map computation
        #output = np.exp(output)
        #output = np.divide(output, output.sum())
        #output = np.divide(output, np.expand_dims(output.sum(1), axis=1))
        target, _ = target.max(0)
        if vid not in scores:
            target_scores[vid] = []
            scores[vid] = []
        scores[vid].append(output[0])
        target_scores[vid].append(target.data.cpu().numpy())
        outputs.append(output[0])
        targets.append(target.data.cpu().numpy()[0])
        correct = target.eq(action.type_as(target)).sum().data.cpu().numpy()
        corr += (100. * correct) / curRGB.size(0)
    #np.savetxt('cmatrix.txt', mtr.value(), fmt="%.2e")
    #print(mapmtr.value())
    #print(mtr.value())
    #plot_confusion_matrix(mtr.value(), [])
    #print(corr/(batch_idx))
    outputs = np.array(outputs)
    targets = np.array(targets)
    print(outputs.shape, targets.shape)
    # Aggregate all of scores, into outputs, targets
    ap = charades_ap(outputs, targets)
    mean_ap = np.mean(ap)
    print('mAP', mean_ap)
    #print('Top5: ', 100*t5cum/(batch_idx))
    f.close()
    floc.close()
    net.train()
    actionClassifier.train()
    return mean_ap, (corr/(batch_idx), 100*t5cum/(batch_idx))

if __name__ == "__main__":
    # Print experiment details
    print_config()
    test()
    train()
