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


class CharadesLoader(data.Dataset):
    def __init__(self, base_dir, input_transform=None, target_transform=None, fps=25, split='train'):
        super(CharadesLoader, self).__init__()
        self.batch_size = 1
        self.fps = 24
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
                s = int(round(float(s)))
                e = int(round(float(e)))
                self.actions[row['id']].append([a, s, e])

    def __getitem__(self, index):
        video_name = self.video_names[index]
        rgb_files = glob(os.path.join(self.base_dir, 'Charades_v1_rgb', video_name, '*'))
        #flow_files = glob(os.path.join(self.base_dir, 'Charades_v1_flow', video_name, '*'))
        seq_len = len(rgb_files) // self.fps - 1
        seq_len = min(64, seq_len) # Cap sequence length
        #h, w = load_img(rgb_files[0]).size
        h = w = 224
        rgb_tensor = torch.Tensor(seq_len, 3, h, w)
        flow_tensor = torch.Tensor(seq_len, 6, h, w)
        #frameNums = [(1+f) * self.fps for f in range(seq_len)]
        target = torch.LongTensor(seq_len, NUM_ACTIONS).zero_()
        for action in self.actions[video_name]:
            a, s, e = action
            for i in range(s, min(e, seq_len)):
                target[i][a] = 1
        if target.sum() == 0:
            return self.__getitem__(index+1)
        for i in range(seq_len):
            frameNum = (1+i) * self.fps
            rgb = load_img(os.path.join(self.base_dir, 'Charades_v1_rgb', video_name, '%s-%06d.jpg' % (video_name, frameNum)))
            rgb = trainImgTransforms(rgb)
            rgb_tensor[i] = rgb
            for flowNum in range(frameNum-1, frameNum+2):
                flowx = load_img(os.path.join(self.base_dir, 'Charades_v1_flow', video_name, '%s-%06dx.jpg' % (video_name, flowNum)))
                flowy = load_img(os.path.join(self.base_dir, 'Charades_v1_flow', video_name, '%s-%06dy.jpg' % (video_name, flowNum)))
                flowx, _, _ = flowx.split()
                flowy, _, _ = flowy.split()
                flowImage = Image.merge("RGB", [flowx,flowy,flowx])
                flowImage = trainFlowTransforms(flowImage)
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
        normalizeFlow = transforms.Normalize(mean=[flowmeanx, flowmeany]*3,
                                     std=[flowstdx, flowstdy]*3)
        rgb_tensor = normalize(rgb_tensor)
        flow_tensor = normalizeFlow(flow_tensor)
        input = (rgb_tensor, flow_tensor)
        input, target = removeEmptyFromTensor(input, target)
        return input, target

    def __len__(self):
        return len(self.video_names)
