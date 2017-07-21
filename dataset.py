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

from utils import *

def load_img(filepath, transforms=None):
    img = Image.open(filepath).convert('RGB')
    if transforms:
        img = self.transform(img)
    return img


def corner_crop_random(img, sz=224):
    w, h = (img.size[0], img.size[1])
    idx = int(np.random.random()*5)
    # 0-3 clockwise starting from top left 4 - center
    crop_coords = (w/2-sz/2, h/2-sz/2, w/2+sz/2, h/2+sz/2)
    if idx == 0:
        crop_coords = (0, 0, sz, sz)
    if idx == 1:
        crop_coords = (w-sz, 0, w, sz)
    if idx == 2:
        crop_coords = (w-sz, h-sz, w, h)
    if idx == 3:
        crop_coords = (0, h-sz, sz, h)
    return img.crop(crop_coords)

CornerCrop = transforms.Lambda(corner_crop_random)

trainImgTransforms = transforms.Compose([
    #transforms.Scale(256),
    #CornerCrop,
    transforms.RandomSizedCrop(224),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

trainFlowTransforms = transforms.Compose([
    transforms.RandomSizedCrop(224),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

Crop = None
if TEST_CROP_MODE == 'CenterCrop':
    Crop = transforms.CenterCrop(224)
elif TEST_CROP_MODE == 'CornerCrop':
    Crop = CornerCrop 
else:
    Crop = transforms.RandomCrop(224)

valTransforms = transforms.Compose([
    transforms.Scale(256),
    Crop,
    transforms.ToTensor()
])

#trainImgTransforms = valTransforms


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
        #split = 'trainval' if split == 'train' else split
        self.video_names = open(os.path.join(base_dir, '%s.txt'%split)).read().split('\n')[:-1]
        #self.video_names = [v.split('/')[-1] for v in glob(os.path.join(self.base_dir, 'Charades_v1_rgb', '*'))]
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.actions = {}
        self.gen_sequence()
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
    def gen_sequence(self):
        self.internal_counter = 0
        self.seq = list(range(len(self.video_names)))
        np.random.shuffle(self.seq)
    
    def load_files(self, files):
        seq_len = len(files)
        h = w = 224
        rgb_tensor = self.load_rgb(files) if USE_RGB else torch.Tensor(seq_len, 3, 1, 1)
        flow_tensor = self.load_flow(files) if USE_FLOW else torch.Tensor(seq_len, 2*NUM_FLOW, 1, 1)
        target = torch.LongTensor(seq_len, NUM_ACTIONS).zero_()
        if self.split in ['train', 'trainval'] and TRAIN_MODE=='SINGLE':
            target = torch.LongTensor(seq_len, 1).fill_(-1)
        
        for i in range(len(files)):
            vid, frameNum = files[i]
            for action in self.actions[vid]:
                a, s, e = action
                # check whether action is present in frameNum
                if s <= frameNum and e >= frameNum:
                    if self.split in ['train', 'trainval'] and TRAIN_MODE=='SINGLE':
                        #cands = all_targets[frameNum].nonzero().cpu().numpy()[0]
                        #target[i] = torch.LongTensor([int(max(cands, key=lambda x: classweights[x]))])
                        #target[i] = torch.LongTensor([int(np.random.choice(cands).astype(int))])
                        if target[i].cpu().numpy() == -1:
                            target[i] = a
                        else:
                            target[i] = target[i] if np.random.random() >= 1./(i+1) else a
                    else:
                        target[i][a] = 1
                
        return (rgb_tensor, flow_tensor), target


    def load_rgb(self, files, h=224, w=224):
        seq_len = len(files)
        rgb_tensor = torch.Tensor(seq_len, 3, h, w)
        for i in range(len(files)):
            vid, frameNum = files[i]
            rgbFileName = os.path.join(self.base_dir, 'Charades_v1_rgb', vid, '%s-%06d.jpg' % (vid, frameNum))
            rgb = load_img(rgbFileName)
            rgb = trainImgTransforms(rgb) if self.split in ['train', 'trainval'] else valTransforms(rgb)
            rgb_tensor[i] = rgb
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        return normalize(rgb_tensor)
    
    
    def load_flow(self, files, h=224, w=224):
        seq_len = len(files)
        flow_tensor = torch.Tensor(seq_len, 2*NUM_FLOW, h, w)
        for i in range(len(files)):
            vid, frameNum = files[i]
            s = NUM_FLOW//2
            e = NUM_FLOW-s
            for flowNum in range(frameNum-s, frameNum+e):
                flowxFileName = os.path.join(self.base_dir, 'Charades_v1_flow', vid, '%s-%06dx.jpg' % (vid, frameNum))
                flowyFileName = os.path.join(self.base_dir, 'Charades_v1_flow', vid, '%s-%06dy.jpg' % (vid, frameNum))
                flowx = load_img(flowxFileName)
                flowy = load_img(flowyFileName)
                flowx, _, _ = flowx.split()
                flowy, _, _ = flowy.split()
                flowImage = Image.merge("RGB", [flowx,flowy,flowx])
                flowImage = trainFlowTransforms(flowImage) if self.split in ['train', 'trainval'] else valTransforms(flowImage)
                flowImage = flowImage[0:2, :, :]
                j = 2*(flowNum - (frameNum-s))
                flow_tensor[i, j:j+2] = flowImage
        flowflatx = (flow_tensor[:, 0, :, :]).contiguous().view(-1)
        flowflaty = (flow_tensor[:, 1, :, :]).contiguous().view(-1)
        flowstdx = torch.std(flowflatx)
        flowmeanx = torch.mean(flowflatx)
        flowstdy = torch.std(flowflaty)
        flowmeany = torch.mean(flowflaty)
        flowstdx = flowstdy = 1.0; flowmeanx = flowmeany = 128/255.0
        normalizeFlow = transforms.Normalize(mean=[flowmeanx, flowmeany]*NUM_FLOW,
                                    std=[flowstdx, flowstdy]* NUM_FLOW)
        return normalizeFlow(flow_tensor)

    def __getitem__(self, index):
        if index == self.__len__() - 1:
            # shuffle dataset
            self.gen_sequence()
        if len(self.remaining) == 0:
            frames = self.get_frame_number_for_vid(self.seq[self.internal_counter])
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
        # TODO: remember to change hardcoded number for final testing
        #return 9347 if self.split == 'train' else len(self.video_names)
        if self.split == 'train':
            return 9347
        elif self.split == 'trainval':
            return 11509
        else:
            return len(self.video_names)
        #return len(self.video_names)
