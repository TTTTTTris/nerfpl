from logging import root
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import random
import cv2
import math
import matplotlib.pyplot as plt
from .plus import sin_plus,cos_plus
from scipy.io import loadmat


class lensless_dataset(Dataset):
    def __init__(self, hparams, split='train'):
        self.hparams = hparams
        self.split = split
        self.read_meta()
        # print(self.hparams.root_dir)
        
    def read_meta(self):      
        # grid
        [x, y] = torch.meshgrid(torch.linspace(-1, 1, self.hparams.ctf_wh[1]), torch.linspace(-1, 1, self.hparams.ctf_wh[0]))
        x = x.type(torch.float32)
        y = y.type(torch.float32)
        x = x.view(1, self.hparams.ctf_wh[1], self.hparams.ctf_wh[0], 1)
        y = y.view(1, self.hparams.ctf_wh[1], self.hparams.ctf_wh[0], 1)
        xy = torch.cat([x, y], dim=-1)     # (1, h, w, 2)
        
        self.rawimg_max = torch.tensor(1.)
        if self.hparams.simulation:    # simulation experiment
            path_RawImgSetMax = os.path.join(self.hparams.root_dir, 'RawImgSetMax.mat')
            RawImgSetMax = torch.tensor((loadmat(path_RawImgSetMax))['RawImgSetMax'], dtype=torch.float32)
            self.rawimg_max = RawImgSetMax[0]
            # print(self.rawimg_max)
        
        if self.split == 'train':
            self.all_rgbs = []  # raw images
            self.all_ctfs = []  # ctfs 
            self.img_h = self.hparams.img_wh[1]
            self.img_w = self.hparams.img_wh[0]            
       
            # shape: (h, w, 8(default))
            ctf_path = os.path.join(self.hparams.root_dir, 'PropCTFSet.mat')
            ctf_set = loadmat(ctf_path)            
            ctfs_all = ctf_set['PropCTFSet']
            ctfs_all = ctfs_all[:,:,0:self.hparams.img_num];
       
            for imgid in range(self.hparams.img_num):            
                img_path = os.path.join(self.hparams.root_dir, 'RawImg_' + str(imgid + 1).zfill(2) + '.png')
                img = ((cv2.imread(img_path, -1)).astype(np.float32))/255.
          
                img = np.reshape(img, [self.img_h, self.img_w, 1])
                img = np.transpose(img, [2,0,1])
                img = torch.tensor(img)   # (1, h, w)
                self.all_rgbs += [img]

                ctf_t = ctfs_all[:, :, imgid]
                ctf = np.zeros([1, self.hparams.ctf_wh[1], self.hparams.ctf_wh[0]], dtype=np.complex64) # (1, h, w) complex64
                ctf[0, :, :] = ctf_t[::-1, ::-1]
                ctf = torch.tensor(ctf)
                self.all_ctfs.append(ctf)
                
            self.all_rgbs = torch.cat(self.all_rgbs, dim=0)          # (8, h, w)
            self.all_rays = xy                                       # (1, h, w, 2)
            self.all_ctfs = torch.cat(self.all_ctfs, dim=0)          # (8, h, w)
        else:
            # GT object
            img_path = os.path.join(self.hparams.root_dir, 'GT_amp.tif')
            img_amp = ((plt.imread(img_path)).astype(np.float32))/255.

            img_path = os.path.join(self.hparams.root_dir, 'GT_phs.tif')
            img_phase = ((plt.imread(img_path)).astype(np.float32))/255.
            img_phase = img_phase * math.pi * self.hparams.pai_scale
            
            input_img = np.concatenate([img_amp[np.newaxis, :,:, np.newaxis], img_phase[np.newaxis, :,:, np.newaxis]], axis=-1)
            input_img = torch.tensor(input_img)     # [1, h, w, 2]

            self.all_rgbs = input_img    # (1, h, w, 2) amp, phase
            self.all_rays = xy           # (1, h, w, 2)

    def __len__(self):
        return self.all_rgbs.shape[0]

    def __getitem__(self, idx):
        if self.split == 'train':
            sample = {'rays': self.all_rays[0, ...],          # [h, w, 2]
                      'rgbs': self.all_rgbs[idx, ...],        # [h, w]
                      'ctfs': self.all_ctfs[idx, ...],        # [h, w]
                      'rmax': self.rawimg_max,                 
                      'zidx': idx
                     }
        else:
            sample = {'rays': self.all_rays[0, ...],          # [h, w, 2]
                      'rgbs': self.all_rgbs[idx, ...]         # [h, w, 2]
                     }
        return sample



def get_mgrid(sideLen, dim=2):
    '''
    Generate a flattened grid of (x, y, ...) coordinate in a range of -1 to 1.
    sideLen:    int
    dim:        int
    '''
    tensors = tuple(dim * [torch.linspace(-1, 1, sideLen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def zero2eps(B, eps=1e-6):
    mask = (B >= 0) * (B < eps)
    B[mask==True] = eps
    mask = (B < 0) * (B > -eps)
    B[mask==True] = -eps
    return B


def operation(A, B, E, operator='add', idx=0):
    if operator == 'fit':
        return torch.stack([A, B], dim=-1)
    if operator == 'add':
        if idx == 0:
            return A + B
        elif idx == 1:
            return A + 2 * B
            # return A + 1.1 * B
            # return A + 1.5 * B
    elif operator == 'sub':
        if idx == 0:
            return A - B
        elif idx == 1:
            return A - 2 * B 
            # return A - 1.1 * B
            # return A - 1.5 * B
    elif operator == 'mul':
        if idx == 0:            
            return (A + 0.5) * (B + 0.5)
        elif idx == 1:            
            return (A + 0.5) * (B + 1)
            # return A * (B + 0.1)
            # return A * (B + 0.5)
    elif operator == 'div':
        eps = 1e-6
        if idx == 0:
            return A / zero2eps(B + 0.5)
        elif idx == 1:
            return A / zero2eps(B + 1)
    elif operator == 'pluscos':
        if idx == 0:            
            return A*torch.cos(torch.pi*B)
        elif idx == 1:
            return (A+1)*torch.cos(torch.pi*B)
        elif idx == 2:
            return A+1
    elif operator == 'plussin':
        if idx == 0:            
            return A*torch.sin(B)
            # return A*torch.sin(torch.pi*B)
        elif idx == 1:
            return (A+1)*torch.sin(B)
            # return (A+1)*torch.sin(torch.pi*B)
        elif idx == 2:
            return A+1
    elif operator == 'pluscosE':
        if idx == 0:            
            return A*torch.cos(torch.pi*B)
        elif idx == 1:
            return E*torch.cos(torch.pi*B)
        elif idx == 2:
            return A+1
    elif operator == 'conv':
        conv = F.conv2d(A.view(1,1,178,178),E)
        return conv.squeeze()
    elif operator == 'sin':
        return torch.sin(B*torch.pi)
        # return torch.sin((B-0.5)*torch.pi)
    elif operator == 'cos':
        return torch.cos(B*torch.pi)
    if operator == 'power_two':
            return B**2
            # return (B-0.5)**2
    elif operator == 'sinx':
        sinx = sin_plus.apply(0.5*torch.pi*B)
        return sinx
    elif operator == 'cosx':
        cosx = cos_plus.apply(torch.pi*B)
        return cosx
    elif operator == 'exp':
        B = 0.5*torch.pi*B
        if idx == 0:            
            return A*torch.complex(torch.cos(B),torch.sin(B))
        elif idx == 1:
            return (A+1)*torch.complex(torch.cos(B),torch.sin(B))
    else:
        raise NameError


class OperatorDataset(Dataset):
    def __init__(self, root_dir, E, operator='add'):
        
        image_path1 = os.path.join(root_dir, 'A.png')
        image_path2 = os.path.join(root_dir, 'B.png')

        # [n, n]
        img1 = (plt.imread(image_path1)).astype(np.float32)
        img2 = (plt.imread(image_path2)).astype(np.float32)

        inverse = False
        if inverse == True:
            img2 = (plt.imread(image_path1)).astype(np.float32)
            img1 = (plt.imread(image_path2)).astype(np.float32)
    

        # ground truth
        self.A = torch.tensor(self.rgb2gray(img1))
        self.B = torch.tensor(self.rgb2gray(img2))
        E = torch.tensor(self.rgb2gray(img1))+1

        # model input
        self.grid1 = get_mgrid(self.A.shape[0], dim=2)
        self.grid2 = get_mgrid(self.B.shape[0], dim=2)

        # observed value 
        self.observations = []
        if operator == 'fit':
            self.observations.append(operation(self.A, self.B, E, operator))
        if operator == 'pluscosE':
            self.observations.append(operation(self.A, self.B, E, operator, idx=0))
            self.observations.append(operation(self.A, self.B, E, operator, idx=1))
            self.observations.append(operation(self.A, self.B, E, operator, idx=2))
        else:
            print(operator)
            self.observations.append(operation(self.A, self.B, E, operator, idx=0))
            self.observations.append(operation(self.A, self.B, E, operator, idx=1))
            # self.observations.append(operation(self.A, self.B, E, operator, idx=2)) # new line
            print(self.observations)
        # print(self.observations[0].size())
        self.observations = torch.stack(self.observations, dim=0) # [x, n, n]
    
    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):
        if idx > self.observations.shape[0]: raise IndexError
        return {
            'grid1':        self.grid1,
            'grid2':        self.grid2,
            'observation':  self.observations[idx],
            'gt1':          self.A,
            'gt2':          self.B
        }

    def rgb2gray(self, rgb):
        return 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]



if __name__ == '__main__':
    print('lensless_data.py')
    root_dir = '../DataSet/data'

    dataset = OperatorDataset(root_dir=root_dir, operator='fit')

    # print(dataset.A.shape)
    # print(dataset.B.shape)
    # print(dataset.observations.shape)
    # print(dataset.grid1.shape)

    c = operation(dataset.A, dataset.B, 'sin', idx=0)
    print(torch.max(c))
    print(torch.min(c))

    # c = operation(dataset.A, dataset.B, 'div', idx=1)
    # print(torch.max(c))
    # print(torch.min(c))

