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
import math

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
    #transforms.RandomHorizontalFlip(),
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
        self.remaining = []
        self.internal_counter = 0
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
                s = int(math.floor(float(s)*self.fps))
                e = int(math.ceil(float(e)*self.fps))
                self.actions[row['id']].append([a, s, e])
    def load_files(self, files):
        seq_len = len(files)
        #rgbFileName = os.path.join(self.base_dir, 'Charades_v1_rgb', files[0][0], '%s-%06d.jpg' % (files[0][0], files[0][1]))
        #rgb = load_img(rgbFileName)
        #h, w = rgb.size
        h = w = 224
        rgb_tensor = torch.Tensor(seq_len, 3, h, w)
        flow_tensor = torch.Tensor(seq_len, 6, h, w)
        target = torch.LongTensor(seq_len, NUM_ACTIONS).zero_()
        if self.split == 'train' and TRAIN_MODE=='SINGLE':
            target = torch.LongTensor(seq_len, 1).fill_(-1)
        
        for i in range(len(files)):
            vid, frameNum = files[i]
            for action in self.actions[vid]:
                a, s, e = action
                # check whether action is present in frameNum
                if s <= frameNum and e >= frameNum:
                    if self.split == "train" and TRAIN_MODE=='SINGLE':
                        #cands = all_targets[frameNum].nonzero().cpu().numpy()[0]
                        #target[i] = torch.LongTensor([int(max(cands, key=lambda x: classweights[x]))])
                        #target[i] = torch.LongTensor([int(np.random.choice(cands).astype(int))])
                        if target[i].cpu().numpy() == -1:
                            target[i] = a
                        else:
                            target[i] = target[i] if np.random.random() >= 1./(i+1) else a
                    else:
                        target[i][a] = 1 
            rgbFileName = os.path.join(self.base_dir, 'Charades_v1_rgb', vid, '%s-%06d.jpg' % (vid, frameNum))
            rgb = load_img(rgbFileName)
            rgb = trainImgTransforms(rgb) if self.split == 'train' else valTransforms(rgb)
            rgb_tensor[i] = rgb
            for flowNum in range(frameNum-1, frameNum+2):
                flowxFileName = os.path.join(self.base_dir, 'Charades_v1_flow', vid, '%s-%06dx.jpg' % (vid, frameNum))
                flowyFileName = os.path.join(self.base_dir, 'Charades_v1_flow', vid, '%s-%06dy.jpg' % (vid, frameNum))
                flowx = load_img(flowxFileName)
                flowy = load_img(flowyFileName)
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
        
        return (rgb_tensor, flow_tensor), target

    def __getitem__(self, index):
        if len(self.remaining) == 0:
            frames = self.get_frame_number_for_vid(self.internal_counter)
            self.remaining = frames 
            self.internal_counter += 1
        last = min(self.batch_size, len(self.remaining))
        frames = self.remaining[:last]
        self.remaining = self.remaining[last:]
        if self.frame_selection == "TEST":
            frames = self.get_frame_number_for_vid(index)
        return self.load_files(frames)
    
    def get_frame_number_for_vid(self, index):
        vid = self.video_names[index]
        rgb_files = glob(os.path.join(self.base_dir, 'Charades_v1_rgb', vid, '*'))
        N = len(rgb_files)
        seq_len = N // self.fps -1
        #seq_len = min(self.batch_size, seq_len)
        # frame selection code
        all_targets = torch.LongTensor(N, NUM_ACTIONS).zero_()
        #frameNums = [(1+f) * self.fps for f in range(seq_len)]
        for action in self.actions[vid]:
            a, s, e = action
            for i in range(s, min(e, N)):
                all_targets[i][a] = 1
        frameNums = []
        valid_frames = list(filter(lambda i: all_targets[i].sum() > 0, range(self.fps, N-self.fps)))
        if all_targets.sum() == 0 or len(valid_frames) == 0:
            return self.get_frame_number_for_vid(index+1)
        if self.frame_selection == 'RANDOM':
            frameNums = random.sample(valid_frames, min(seq_len, len(valid_frames)))
            frameNums.sort()
        elif self.frame_selection == 'TEST':
            frameNums = [int(ii) for ii in np.linspace(2, N-25-1, self.testGAP)]
        else:
            frameNums = findClosestFrames(valid_frames, self.fps, N-self.fps, self.fps)
            frameNums = frameNums if len(frameNums) <= seq_len else frameNums[:seq_len]
        return list(zip([vid]*len(frameNums), frameNums))

    def __getitem_old__(self, index):
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
        seq_len = min(len(frameNums), self.batch_size) # Cap sequence length
        target = torch.LongTensor(seq_len, NUM_ACTIONS).zero_()
        if self.split == 'train' and TRAIN_MODE=='SINGLE':
            target = torch.LongTensor(seq_len, 1).zero_()
        rgb_tensor = torch.Tensor(seq_len, 3, h, w)
        flow_tensor = torch.Tensor(seq_len, 6, h, w)
        for i in range(len(frameNums)):
            frameNum = frameNums[i]#(1+i) * self.fps
            if self.split == "train" and TRAIN_MODE=='SINGLE':
                cands = all_targets[frameNum].nonzero().cpu().numpy()[0]
                #target[i] = torch.LongTensor([int(max(cands, key=lambda x: classweights[x]))])
                target[i] = torch.LongTensor([int(np.random.choice(cands).astype(int))])
            else:
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
    
    def randomSamplesOld(self, seq_len):
        h = w = 224
        rgb_tensor = torch.Tensor(seq_len, 3, h, w)
        flow_tensor = torch.Tensor(seq_len, 6, h, w)        
        targets = torch.LongTensor(seq_len, NUM_ACTIONS).zero_()
        if TRAIN_MODE=='SINGLE':
            targets = torch.LongTensor(seq_len, 1).zero_()
        for i in range(seq_len):
            vid = random.randint(0, len(self.video_names)-1)
            input, target = self.__getitem__(vid)
            frame = random.randint(0, len(target)-1)
            rgb_tensor[i] = input[0][frame]
            flow_tensor[i] = input[1][frame]
            targets[i] = target[frame]
        input = (rgb_tensor, flow_tensor)
        return input, targets

    def randomSamples(self, seq_len):
        frames = []
        for i in range(seq_len):
            vid = random.randint(0, len(self.video_names)-1)
            frames_vid = self.get_frame_number_for_vid(vid)
            frame = frames_vid[random.randint(0, len(frames_vid)-1)]
            frames.append(frame)
        return self.load_files(frames) 
    
    def __len__(self):
        # The number of size 32 batches
        return 9347 if self.split == 'train' else len(self.video_names)
        #return len(self.video_names)
