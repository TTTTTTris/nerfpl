from ast import operator
from asyncio import tasks
import os, sys
from re import L
from types import ModuleType
from opt import get_opts
import torch
from torch import nn
from torch import fft
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.lensless_data import operation

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2

# models
from models.nerf import Embedding, NeRF, Siren, MLP
from models.rendering import render_rays_lensless_2stream
from models.calc_ctf import CTF

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.logging import TestTubeLogger
from math import pi

import numpy as np
import pandas as pd
import csv

import time
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path) 


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams = hparams

        self.loss = loss_dict[hparams.loss_type]()

        if self.hparams.modelType == 'SIREN':
            self.model1 = Siren(in_features=2, 
                        out_features=1, 
                        hidden_layers=self.hparams.hidden_layers, 
                        hidden_features=self.hparams.hidden_features, 
                        outermost_linear=True)
            self.model2 = Siren(in_features=2, 
                        out_features=1, 
                        hidden_layers=self.hparams.hidden_layers, 
                        hidden_features=self.hparams.hidden_features, 
                        outermost_linear=True)
            # self.model3 = Siren(in_features=2, 
            #             out_features=1, 
            #             hidden_layers=self.hparams.hidden_layers, 
            #             hidden_features=self.hparams.hidden_features, 
            #             outermost_linear=True)
        elif self.hparams.modelType == 'MLP':
            self.model1 = MLP(in_channels_xy=2, D=self.hparams.hidden_layers, W=self.hparams.hidden_features)
            self.model2 = MLP(in_channels_xy=2, D=self.hparams.hidden_layers, W=self.hparams.hidden_features)            
        elif self.hparams.modelType == 'MLP+PE':
            self.freq_A = 20
            self.freq_B = 20
            # self.freq_C = 20 # pluscosE
            self.embedding_A = Embedding(2, self.freq_A) 
            self.embedding_B = Embedding(2, self.freq_B) 
            # self.embedding_C = Embedding(2, self.freq_C) # pluscosE
            self.embeddings = [self.embedding_A, self.embedding_B]
            # self.embeddings = [self.embedding_A, self.embedding_B, self.embedding_C] # pluscosE
            self.model1 = MLP(in_channels_xy=self.freq_A*2*2+2, D=self.hparams.hidden_layers, W=self.hparams.hidden_features)
            self.model2 = MLP(in_channels_xy=self.freq_B*2*2+2, D=self.hparams.hidden_layers, W=self.hparams.hidden_features)
            # self.model3 = MLP(in_channels_xy=self.freq_B*2*2+2, D=self.hparams.hidden_layers, W=self.hparams.hidden_features) # pluscosE

       
        self.models = [self.model1, self.model2]
        # self.models = [self.model1, self.model2, self.model3] # pluscosE
        self.batch_cnt = 0
                
    def decode_train_batch(self, batch):
        grid1 = batch['grid1']
        grid2 = batch['grid2']
        observation = batch['observation']
        gt1 = batch['gt1']
        gt2 = batch['gt2']
        return grid1, grid2, observation, gt1, gt2

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B, num, chans = rays.shape # [batch, kernel_h, kernel_w, 2]
        rays = rays.view(-1,chans)
        B = rays.shape[0]       
        results_amp = []
        results_phase = []
        for i in range(0, B, self.hparams.chunk):
            rays_chunk = rays[i:i+self.hparams.chunk]
        
            if self.hparams.modelType == 'SIREN':    
                res1, _ = self.models[0](rays_chunk)
                res2, _ = self.models[1](rays_chunk)
            elif self.hparams.modelType == 'MLP':  
                res1 = self.models[0](rays_chunk)
                res2 = self.models[1](rays_chunk)
            elif self.hparams.modelType == 'MLP+PE':
                res1 = self.models[0](self.embeddings[0](rays_chunk))
                res2 = self.models[1](self.embeddings[1](rays_chunk))
                # res2 = self.models[2](self.embeddings[2](rays_chunk)) # pluscosE

            results_amp.append(res1)
            results_phase.append(res2)

        results_amp = torch.cat(results_amp, dim=0)
        results_phase = torch.cat(results_phase, dim=0)
        return results_amp, results_phase

    def prepare_data(self):     # OK
        dataset = dataset_dict['operator']
        self.train_dataset = dataset(self.hparams.root_dir, 0, self.hparams.operator)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        self.scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [self.scheduler]

    def train_dataloader(self):         # OK
        return DataLoader(self.train_dataset,
                          shuffle = False,
                          num_workers = 4,
                          batch_size = self.hparams.batch_size,
                          pin_memory = True)
    
    
    def training_step(self, batch, batch_idx):
        log = {'lr': get_learning_rate(self.optimizer)}
                
        grid1, grid2, observation, gt1, gt2 = self.decode_train_batch(batch)

        # [1, 178^2, 2]
        # print(grid1.shape)
        # image_pred1, image_pred2 = self(grid1)          # [batch, h, w]
        image_pred1, image_pred2, image_pred3 = self(grid1)          # pluscosE

        # pred object
        w = observation.shape[1]
        h = observation.shape[2]

        # 'fit' is special 

        image_pred1 = image_pred1.view(-1, w, h)
        image_pred2 = image_pred2.view(-1, w, h)
        image_pred3 = image_pred3.view(-1, w, h) # pluscosE

        # operation
        # observation_pred = operation(image_pred1, image_pred2, 0, self.hparams.operator, idx=batch_idx)
        observation_pred = operation(image_pred1, image_pred2, image_pred3, self.hparams.operator, idx=batch_idx) # pluscosE

        ## A
        # flag_origin2A = True
        # flag_origin2B = False

        # # B
        # flag_origin2A = False
        # flag_origin2B = True

        ## 
        flag_origin2A = False
        flag_origin2B = False

        if flag_origin2A == True:
            if self.hparams.operator == 'add': 
                # loss = ((C-B) - A)^2
                if batch_idx == 0:
                    image1_reconstruct = observation[0] - image_pred2
                elif batch_idx == 1:
                    image1_reconstruct = observation[0] - image_pred2 * 2
            elif self.hparams.operator == 'sub': 
                if batch_idx == 0:
                    image1_reconstruct = observation[0] + image_pred2
                elif batch_idx == 1:
                    image1_reconstruct = observation[0] + image_pred2 * 2
            elif self.hparams.operator == 'mul': 
                if batch_idx == 0:
                    image1_reconstruct = observation[0] / (image_pred2 + 0.5) - 0.5
                elif batch_idx == 1:
                    image1_reconstruct = observation[0] / (image_pred2 + 1) - 0.5

            elif self.hparams.operator == 'div':
                # loss = ((B_pred * A/B) - A_pred)^2
                # image1_reconstruct = operation(observation[0], image_pred2, 'mul', idx=batch_idx)
                if batch_idx == 0:
                    image1_reconstruct = observation[0] * (image_pred2 + 0.5)
                elif batch_idx == 1:
                    image1_reconstruct = observation[0] * (image_pred2 + 1)

            log['train/loss'] = loss = self.loss(image1_reconstruct, image_pred1)       

        elif flag_origin2B == True:
            # todo
            if self.hparams.operator == 'add': 
                # loss = ((C-A) - B)^2
                if batch_idx == 0:
                    image2_reconstruct = observation[0] - image_pred1
                elif batch_idx == 1:
                    image2_reconstruct = (observation[0] - image_pred1) / 2
            elif self.hparams.operator == 'sub': 
                if batch_idx == 0:
                    image2_reconstruct = image_pred1 - observation[0]
                elif batch_idx == 1:
                    image2_reconstruct = (image_pred1 - observation[0]) / 2
            elif self.hparams.operator == 'mul': 
                if batch_idx == 0:
                    image2_reconstruct = observation[0] / (image_pred1 + 0.5) - 0.5
                elif batch_idx == 1:
                    image2_reconstruct = observation[0] / (image_pred1 + 0.5) - 1
            # if self.hparams.operator == 'sin': 
            #     # image2_reconstruct = torch.asin(observation[0])
            #     image2_reconstruct = torch.asin(observation[0])/torch.pi+0.5
            # if self.hparams.operator == 'cos': 
            #     image2_reconstruct = torch.acos(observation[0])/torch.pi
            # if self.hparams.operator == 'power_two': 
            #     image2_reconstruct = torch.sqrt(observation[0]) # +0.5
            log['train/loss'] = loss = self.loss(image2_reconstruct, image_pred2)       
        else:
            # loss
            # print(observation_pred.dtype)
            if observation_pred.dtype == torch.float32:
                if batch_idx == 2:
                    log['train/loss'] = loss =  self.loss(observation_pred, image_pred1+1) 
                else:
                    log['train/loss'] = loss =  self.loss(observation_pred, observation) 
            else:
                # real+imag
                # loss1 = self.loss(torch.real(observation_pred), torch.real(observation))
                # loss2 = self.loss(torch.imag(observation_pred), torch.imag(observation))
                # log['train/loss'] = loss =  loss1+loss2
                # mod
                log['train/loss'] = loss =  self.loss(torch.abs(observation_pred), torch.abs(observation))

        # print(batch_idx, torch.max(observation))

        # psnr
        with torch.no_grad():
            psnr1 = psnr(image_pred1, gt1)
            psnr2 = psnr(image_pred2, gt2)
            log['train/psnr1'] = psnr1
            log['train/psnr2'] = psnr2

        if self.batch_cnt == 0 or self.batch_cnt % 1000 == 999:
            fig, axes = plt.subplots(1, 6, figsize=(18,18)) # (w,h)
            axes[0].imshow(gt1.cpu().view(w,h).detach().numpy())
            axes[0].set_title('gt1')
            axes[1].imshow(image_pred1.cpu().view(w,h).detach().numpy())
            axes[1].set_title('psnr: %0.2f'%(psnr1))

            axes[2].imshow(gt2.cpu().view(w,h).detach().numpy())
            axes[2].set_title('gt2')
            axes[3].imshow(image_pred2.cpu().view(w,h).detach().numpy())
            axes[3].set_title('psnr: %0.2f'%(psnr2))

            if self.hparams.operator is not 'fit':
                axes[4].imshow(observation.cpu().view(w,h).detach().numpy())
                axes[4].set_title('gt1 ' + self.hparams.operator + ' gt2')
                axes[5].imshow(observation_pred.cpu().view(w,h).detach().numpy())
                axes[5].set_title('loss: %0.6f'%(loss))
                num = 2
            else:
                axes[4].imshow(gt1.cpu().view(w,h).detach().numpy())
                axes[4].set_title('gt1')
                axes[5].imshow(image_pred1.cpu().view(w,h).detach().numpy())
                axes[5].set_title('loss_total: %0.6f'%(loss))
                num = 1
                    
            path_dir = os.path.join(self.hparams.root_dir, 'image_out', self.hparams.exp_name)
            save_path = '%s/%s_%dx%d_epochs_%06d_%s.png'%(
                path_dir, 
                self.hparams.modelType, 
                self.hparams.hidden_layers, 
                self.hparams.hidden_features, 
                self.batch_cnt/num, 
                self.hparams.operator)

            plt.savefig(save_path)
            print(save_path)
            # print('saved!')
            # plt.show()
            plt.close()

        self.batch_cnt = self.batch_cnt + 1

        return {'loss': loss*1000,
                'progress_bar': {'train_psnr': psnr2},
                'log': log
               }

    
  
if __name__ == '__main__':
    hparams = get_opts()
    hparams.simulation = True

    flag_debug = False

    # 
    hparams.hidden_layers = 4
    hparams.hidden_features = 256

    #
    hparams.operator    = 'sinx' # fit, add, sub, mul, div, ln, etc 
    hparams.modelType   = 'MLP+PE' # MLP, MLP+PE, SIREN

    # hparams.exp_name    = 'exp_lr0.001'
    # hparams.exp_name    = 'exp_lr0.001_origin2A'
    hparams.exp_name    = 'exp_lr0.001'
    if flag_debug == True:
        hparams.exp_name    = 'debug'
    hparams.exp_name    = hparams.exp_name + '_' + hparams.modelType + '_' + hparams.operator # 'pi_'

    # hparams.task_dir    = '/data/liuzhen/meta_learning_data/DataSets/operator_tasks_celea_face_100_178/' 
    
    hparams.num_epochs  = 10000

    ####
    if flag_debug == True:
        tasks = 1
        record = False
    else:
        tasks = 10
        record = True
    ####
    for idx in range(tasks):
        # hparams.root_dir = os.path.join(hparams.task_dir, 'task_' + str(idx).zfill(6))

        image_out_path = os.path.join(hparams.root_dir, 'image_out', hparams.exp_name)
        mkdir(image_out_path) 
    
        system = NeRFSystem(hparams)
        checkpoint_callback = ModelCheckpoint(filepath=os.path.join(hparams.root_dir, f'ckpts/{hparams.exp_name}', '{epoch:d}'),
                                                                    monitor='train/loss',
                                                                    mode='min',
                                                                    save_top_k=10)
        logger = TestTubeLogger(
            save_dir=os.path.join(hparams.root_dir, f'logs'),
            name=hparams.exp_name,
            debug=False,
            create_git_tag=False
        )
        # print(os.path.join(hparams.root_dir, f'logs'))
        
        trainer = Trainer(max_epochs=hparams.num_epochs,
                        checkpoint_callback=checkpoint_callback,
                        resume_from_checkpoint=hparams.ckpt_path,
                        logger=logger,
                        early_stop_callback=None,
                        weights_summary=None,
                        progress_bar_refresh_rate=1,
                        gpus=hparams.num_gpus,
                        distributed_backend='ddp' if hparams.num_gpus>1 else None,
                        num_sanity_val_steps=1,
                        benchmark=True,
                        amp_level = 'O2',
                        profiler=hparams.num_gpus==1,
                        log_every_n_steps=1,
                        )# gradient_clip_val=1 if hparams.operator == 'div' else 0)

        trainer.fit(system)

        # get the max psnr of image A and B 
        # /data/liuzhen/meta_learning_data/DataSets/operator_tasks_celea_face_100_178/task_000000/logs/SIREN_add_exp_1/version_0/
        metrics_path = os.path.join(hparams.root_dir, 'logs', hparams.exp_name, 'version_0', 'metrics.csv')
        df = pd.read_csv(metrics_path)
        print(df.info(2))
        maxPsnrA = df['train/psnr1'].max()
        maxPsnrB = df['train/psnr2'].max()
        print(maxPsnrA, maxPsnrB)

        if record == True:
            # record the max psnr of A and B
            log_path = os.path.join('./logs')
            mkdir(log_path)
            # eg: ./logs/SIREN_fit_exp_1.csv
            file_path = os.path.join(log_path, hparams.exp_name + '.csv')
            with open(file_path, 'a+', encoding='utf-8', newline='') as file:
                rows = len((open(file_path)).readlines())
                
                fieldnames = ['task', 'psnrA', 'psnrB']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if rows == 0:
                    writer.writeheader()
                writer.writerow({'task': idx, 'psnrA': maxPsnrA, 'psnrB': maxPsnrB})