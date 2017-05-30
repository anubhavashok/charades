import torch
import torch.utils.data as data
from torchvision import transforms

import os
from os import listdir
from os.path import join
#from PIL import Image
import cv2
import csv
from glob import glob

def load_img(filepath):
    #img = Image.open(filepath).convert('YCbCr')
    #y, _, _ = img.split()
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float')/255.0
    img = cv2.resize(img, (224, 224))
    #img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return img


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
        h, w, _ = load_img(rgb_files[0]).shape
        #h = w = 224
        rgb_tensor = torch.Tensor(seq_len, 3, h, w)
        flow_tensor = torch.Tensor(seq_len, 6, h, w)
        #frameNums = [(1+f) * self.fps for f in range(seq_len)]
        for i in range(seq_len):
            frameNum = (1+i) * self.fps
            rgb = load_img(os.path.join(self.base_dir, 'Charades_v1_rgb', video_name, '%s-%06d.jpg' % (video_name, frameNum)))
            #rgb = self.input_transform(rgb)
            rgb_tensor[i, 0, :, :] = torch.from_numpy(rgb[:, :, 0])
            rgb_tensor[i, 1, :, :] = torch.from_numpy(rgb[:, :, 1])
            rgb_tensor[i, 2, :, :] = torch.from_numpy(rgb[:, :, 2])
            for flowNum in range(frameNum-1, frameNum+2):
                flowx = load_img(os.path.join(self.base_dir, 'Charades_v1_flow', video_name, '%s-%06dx.jpg' % (video_name, flowNum)))
                flowy = load_img(os.path.join(self.base_dir, 'Charades_v1_flow', video_name, '%s-%06dy.jpg' % (video_name, flowNum)))
                flowx = flowx[:, :, 0]#cv2.cvtColor(flowx, cv2.COLOR_BGR2GRAY)
                flowy = flowy[:, :, 0]#cv2.cvtColor(flowy, cv2.COLOR_BGR2GRAY)
                #flowx = self.input_transform(flowx)
                #flowy = self.input_transform(flowy)
                j = 2*(flowNum - (frameNum-1))
                flow_tensor[i, j, :, :] = torch.from_numpy(flowx)
                flow_tensor[i, j+1, :, :] = torch.from_numpy(flowy)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        normalizeFlow = transforms.Normalize(mean=[0.485],
                                     std=[0.229])
        rgb_tensor = normalize(rgb_tensor)
        flow_tensor = normalizeFlow(flow_tensor)
        input = (rgb_tensor, flow_tensor)
        target = [157] * seq_len
        for action in self.actions[video_name]:
            a, s, e = action
            for i in range(s, min(e, seq_len)):
                target[i] = a
        target = torch.Tensor(target)
        #if self.input_transform:
        #    input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return input, target

    def __len__(self):
        return len(self.video_names)
