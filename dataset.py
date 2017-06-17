import torch
import torch.utils.data as data
from torchvision import transforms

import os
from os import listdir
from os.path import join
from PIL import Image
import cv2
import csv
from glob import glob
import numpy as np
import random

from config import *
from utils import *

def load_img(filepath, transforms=None):
    img = Image.open(filepath).convert('RGB')
    if transforms:
        img = self.transform(img)
    return img

trainImgTransforms = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

trainFlowTransforms = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

Crop = None
if TEST_CROP_MODE == 'CenterCrop':
    Crop = transforms.CenterCrop
else:
    Crop = transforms.RandomCrop

valTransforms = transforms.Compose([
    transforms.Scale(256),
    Crop(224),
    transforms.ToTensor()
])

def resizeAndCrop(img, sz):
    # First resize image so that smallest dim is at least equal to sz
    h, w, _ = img.shape
    '''
    if h < sz:
        ratio = sz/h
        ns = (int(ratio*w)+1, int(ratio*h)+1)
        img = cv2.resize(img, ns)
    h, w, _ = img.shape
    if w < sz:
        ratio = sz/w
        ns = (int(ratio*w)+1, int(ratio*h)+1)
        img = cv2.resize(img, ns)
    '''
    ratio = sz/h if h <= w else sz/w
    ns = (int(ratio*w)+1, int(ratio*h)+1)
    img = cv2.resize(img, ns)
    h, w, _ = img.shape
    # pick a random point to crop a sz x sz square
    st = random.randint(0, h-224) if st==None else st
    en = random.randint(0, w-224) if en==None else en
    cropped = img[st:st+224, en:en+224, :]
    return cropped

class CharadesLoader(data.Dataset):
    def __init__(self, base_dir, input_transform=None, target_transform=None, fps=24, split='train', frame_selection='SPACED', batch_size=32):
        super(CharadesLoader, self).__init__()
        self.testGAP = 25
        self.frame_selection = frame_selection
        self.split = split
        self.batch_size = batch_size
        self.fps = fps
        self.base_dir = base_dir
        self.video_names = open(os.path.join(base_dir, '%s.txt'%split)).read().split('\n')[:-1]
        #self.video_names = [v.split('/')[-1] for v in glob(os.path.join(self.base_dir, 'Charades_v1_rgb', '*'))]
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.actions = {}
        f = open(os.path.join(base_dir, 'vu17_charades', 'Charades_vu17_%s.csv'%split))
        reader = csv.DictReader(f)
        for row in reader:
            self.actions[row['id']] = [] 
            for action in row['actions'].split(';'):
                if action == '':
                    continue
                a, s, e = action.split(' ') 
                a = int(a[1:])
                s = int(float(s)*self.fps)
                e = int(float(e)*self.fps)
                self.actions[row['id']].append([a, s, e])

    def __getitem__(self, index):
        video_name = self.video_names[index]
        rgb_files = glob(os.path.join(self.base_dir, 'Charades_v1_rgb', video_name, '*'))
        N = len(rgb_files)
        seq_len = N // self.fps -1
        seq_len = min(self.batch_size, seq_len) # Cap sequence length
        h = w = 224
        all_targets = torch.LongTensor(N, NUM_ACTIONS).zero_()
        #frameNums = [(1+f) * self.fps for f in range(seq_len)]
        for action in self.actions[video_name]:
            a, s, e = action
            for i in range(s, min(e, N)):
                all_targets[i][a] = 1
        frameNums = []
        valid_frames = list(filter(lambda i: all_targets[i].sum() > 0, range(self.fps, N-self.fps)))
        if all_targets.sum() == 0 or len(valid_frames) == 0:
            return self.__getitem__(index+1)
        if self.frame_selection == 'RANDOM':
            frameNums = random.sample(valid_frames, min(seq_len, len(valid_frames)))
            frameNums.sort()
        elif self.frame_selection == 'TEST':
            frameNums = [int(ii) for ii in np.linspace(2, N-25-1, self.testGAP)]
        else:
            frameNums = findClosestFrames(valid_frames, self.fps, N-self.fps, self.fps)
            frameNums = frameNums if len(frameNums) <= seq_len else frameNums[:seq_len]
        print(frameNums)
        seq_len = min(len(frameNums), self.batch_size) # Cap sequence length
        target = torch.LongTensor(seq_len, NUM_ACTIONS).zero_()
        rgb_tensor = torch.Tensor(seq_len, 3, h, w)
        flow_tensor = torch.Tensor(seq_len, 6, h, w)
        for i in range(len(frameNums)):
            frameNum = frameNums[i]#(1+i) * self.fps
            target[i] = all_targets[frameNum]
            rgb = load_img(os.path.join(self.base_dir, 'Charades_v1_rgb', video_name, '%s-%06d.jpg' % (video_name, frameNum)))
            rgb = trainImgTransforms(rgb) if self.split == 'train' else valTransforms(rgb)
            rgb_tensor[i] = rgb
            for flowNum in range(frameNum-1, frameNum+2):
                flowx = load_img(os.path.join(self.base_dir, 'Charades_v1_flow', video_name, '%s-%06dx.jpg' % (video_name, flowNum)))
                flowy = load_img(os.path.join(self.base_dir, 'Charades_v1_flow', video_name, '%s-%06dy.jpg' % (video_name, flowNum)))
                flowx, _, _ = flowx.split()
                flowy, _, _ = flowy.split()
                flowImage = Image.merge("RGB", [flowx,flowy,flowx])
                flowImage = trainFlowTransforms(flowImage) if self.split == 'train' else valTransforms(flowImage)
                flowImage = flowImage[0:2, :, :]
                j = 2*(flowNum - (frameNum-1))
                flow_tensor[i, j:j+2] = flowImage
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        flowflatx = (flow_tensor[:, 0, :, :]).contiguous().view(-1)
        flowflaty = (flow_tensor[:, 1, :, :]).contiguous().view(-1)
        flowstdx = torch.std(flowflatx)
        flowmeanx = torch.mean(flowflatx)
        flowstdy = torch.std(flowflaty)
        flowmeany = torch.mean(flowflaty)
        flowstdx = flowstdy = 1.0; flowmeanx = flowmeany = 128/255.0
        normalizeFlow = transforms.Normalize(mean=[flowmeanx, flowmeany]*3,
                                     std=[flowstdx, flowstdy]*3)
        rgb_tensor = normalize(rgb_tensor)
        flow_tensor = normalizeFlow(flow_tensor)
        input = (rgb_tensor, flow_tensor)
        #input, target = removeEmptyFromTensor(input, target)
        return input, target
    
    def randomSamples(self, seq_len):
        h = w = 224
        rgb_tensor = torch.Tensor(seq_len, 3, h, w)
        flow_tensor = torch.Tensor(seq_len, 6, h, w)        
        targets = torch.LongTensor(seq_len, NUM_ACTIONS).zero_()
        for i in range(seq_len):
            vid = random.randint(0, len(self.video_names)-1)
            input, target = self.__getitem__(vid)
            frame = random.randint(0, len(target)-1)
            rgb_tensor[i] = input[0][frame]
            flow_tensor[i] = input[1][frame]
            targets[i] = target[frame]
        input = (rgb_tensor, flow_tensor)
        return input, targets
    
    def __len__(self):
        return len(self.video_names)
