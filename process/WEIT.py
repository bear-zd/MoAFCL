import sys, time, os
from typing import OrderedDict
from model.Nets import WeITAdapter

import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from utils import *
import quadprog
sys.path.append('..')
import torch.nn.functional as F
import torch.nn as nn
import utils.cliputils as clu
import itertools
from torch.utils.data import TensorDataset, DataLoader

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return
class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self,adapter_model, tr_dataloader,nepochs=5, lr=0.001, lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=100,
                 args=None):
        self.model: WeITAdapter = adapter_model
        self.model_old: WeITAdapter = adapter_model
        self.fisher = None
        self.nepochs = nepochs
        self.tr_dataloader = tr_dataloader
        self.lr = lr
        self.lr_min = lr_min * 1 / 3
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_decay = args.lr_decay
        self.optim_type = args.optim
        self.clipgrad = clipgrad
        self.args = args
        self.e_rep = 1
        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.old_task=-1
        self.pre_weight = {
            'weight':[],
            'aw':[],
            'mask':[]
        }
        self.device = args.device
        self.feature_data = []
    def extract_feature(self):
        def hook(module, input, output):
            self.feature_data.append(output.permute(1,0,2).detach().cpu().numpy())
        return hook
    def preprocess(self):
        temp = list(itertools.chain.from_iterable(self.feature_data))
        temp = np.stack(temp, axis = 0)
        temp = np.mean(temp, axis=1)
        return temp
    def set_sw(self,glob_weights):
        i = 0
        keys = [k for k, _ in self.model.named_parameters()]
        if len(glob_weights)>0:
            all_weights = []
            for name,para in self.model.named_parameters():
                if 'sw' in name:
                    all_weights.append(glob_weights[i])
                    i=i+1
                else:
                    all_weights.append(para)
            model_dict = self.model.state_dict()
            feature_dict = zip(keys, all_weights)
            # last_dict = OrderedDict({k: torch.Tensor(v) for k, v in zip(last_keys,last_para)})
            save_model = OrderedDict({k: v for k, v in feature_dict})
            state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)
    def get_sw(self):
        sws = []
        for name,para in self.model.named_parameters():
            if 'sw' in name:
                sws.append(para)
        return sws
    def set_trData(self,tr_dataloader):
        self.tr_dataloader = tr_dataloader

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        if "SGD" in self.optim_type:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.lr_decay)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.lr_decay)
        return optimizer

    def train(self,clip_model, t,from_kbs,know):
        if t!=self.old_task:
            self.old_task=t
        lr = self.lr
        for name, para in self.model.named_parameters():
            para.requires_grad = True
        self.model.set_knowledge(t,from_kbs)
        self.optimizer = self._get_optimizer()
        self.model.to(self.device)
        # Loop epochs 
        self.temp_hook = clip_model.model.visual.transformer.resblocks[0].register_forward_hook(self.extract_feature())
        for image,_,_ in self.tr_dataloader:
            clip_model.model.encode_image(image.to(self.device))
        self.temp_hook.remove()
        for e in range(self.nepochs):
            # Train
            loss = self.train_epoch(clip_model, t)
            if e % self.e_rep == self.e_rep -1:
                train_acc = self.eval(clip_model, t)
                print('| Epoch {:3d} | Train loss={:3.1f} acc={:5.1f}% | \n'.format(
                    e + 1, loss ,100 * train_acc), end='')
        if len(self.pre_weight['aw'])<=t:
            self.pre_weight['aw'].append([])
            self.pre_weight['mask'].append([])
            self.pre_weight['weight'].append([])
            for name,para in self.model.named_parameters():
                if 'aw' in name:
                    aw = para.detach()
                    aw.requires_grad = False
                    self.pre_weight['aw'][-1].append(aw)
                elif 'mask' in name:
                    mask = para.detach()
                    mask.requires_grad = False
                    self.pre_weight['mask'][-1].append(mask)
            self.pre_weight['weight'][-1] = self.model.get_weights()
        else:
            self.pre_weight['aw'].pop()
            self.pre_weight['mask'].pop()
            self.pre_weight['weight'].pop()
            self.pre_weight['aw'].append([])
            self.pre_weight['mask'].append([])
            self.pre_weight['weight'].append([])
            for name, para in self.model.named_parameters():
                if 'aw' in name:
                    self.pre_weight['aw'][-1].append(para)
                elif 'mask' in name:
                    self.pre_weight['mask'][-1].append(para)
            self.pre_weight['weight'][-1] = self.model.get_weights()
        self.feature_data = []

        return self.get_sw(), loss, train_acc

    def train_epoch(self,clip_model, t):
        self.model.train()
        # clip_model.model.train()
        total_loss = 0
        total_nums = 0
        clip_model.model.eval()
        client_domain_feature = torch.tensor(self.preprocess(), dtype=torch.float)
        list_image_domain_features = list(DataLoader(TensorDataset(client_domain_feature),batch_size=50, shuffle=False))

        for index, batch in enumerate(self.tr_dataloader):
            image, _ ,label = batch
            image, label = image.to(self.device), label.to(self.device)
            # Forward current model
            
            self.optimizer.zero_grad()
            self.model.zero_grad()
            domain_feature = self.model(list_image_domain_features[index][0].to(self.device))
            mean_domain_feature = domain_feature.mean(dim=0, keepdim=True)
            _mean_domain_features = mean_domain_feature.repeat_interleave(len(clip_model.labels), dim=0)
            text_features = clip_model._get_text_features(_mean_domain_features.half())

            image_features = clip_model.model.encode_image(image).half()
            # image_features_att = self.model(image_features)
            # image_features = torch.mul(image_features_att, image_features)
            # text_features = clip_model.model.encode_text(text).float()

            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            logit_scale = clip_model.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(logits_per_image, label)
            # logits_per_text = logits_per_image.t()
            
            # ground_truth = torch.arange(len(image), dtype=torch.long, device=self.device)
            # loss = (self.ce(logits_per_image, ground_truth) + 
            #     self.ce(logits_per_text, ground_truth))/2
            loss += self.get_loss(t)
            ## 根据这个损失计算梯度，变换此梯度
            total_loss+= loss
            total_nums += len(image)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return total_loss/total_nums
    def l2_loss(self,para):
        return torch.sum(torch.pow(para,2))/2
        
    def get_loss(self,t):
        i = 0
        weight_decay = 0
        sparseness = 0
        approx_loss = 0
        sw = None
        aw = None
        mask = None
        for name,para in self.model.named_parameters():
            if 'sw' in name:
                sw = para
            elif 'aw' in name:
                aw = para
            elif 'mask' in name:
                mask = para
            elif 'atten' in name:
                weight_decay += self.args.wd * self.l2_loss(aw)
                weight_decay += self.args.wd * self.l2_loss(mask)
                sparseness += self.args.lambda_l1 * torch.sum(torch.abs(aw))
                sparseness += self.args.lambda_mask * torch.sum(torch.abs(mask))
                if torch.isnan(weight_decay).sum() > 0:
                    print('weight_decay nan')
                if torch.isnan(sparseness).sum() > 0:
                    print('sparseness nan')
                if t == 0:
                    weight_decay += self.args.wd * self.l2_loss(sw)
                else:
                    for tid in range(t):
                        prev_aw = self.pre_weight['aw'][tid][i]
                        prev_mask = self.pre_weight['mask'][tid][i]
                        m = torch.nn.Sigmoid()
                        g_prev_mask = m(prev_mask)
                        #################################################
                        sw2 = sw.transpose(0,-1)
                        restored = (sw2 * g_prev_mask).transpose(0,-1) + prev_aw

                        a_l2 = self.l2_loss(restored - self.pre_weight['weight'][tid][i])
                        approx_loss += self.args.lambda_l2 * a_l2
                        #################################################
                    i+=1
        loss = weight_decay+sparseness+approx_loss
        return loss

    def eval(self, clip_model,t,train=True):
        self.model.eval()
        clip_model.model.eval()
        dataloaders = self.tr_dataloader
        # texts = clip_model.labels
        # text_features = clu.get_text_features_list(texts, clip_model.model, self.device).float()
        client_domain_feature = torch.tensor(self.preprocess(), dtype=torch.float)
        list_image_domain_features = list(DataLoader(TensorDataset(client_domain_feature),batch_size=50, shuffle=False))

        total = 0
        correct = 0
        # Loop batches
        with torch.no_grad():
            for index, batch in enumerate(dataloaders):
                image, _, label = batch
                image, label = image.to(self.device), label.to(self.device)
                domain_feature = self.model(list_image_domain_features[index][0].to(self.device))
                mean_domain_feature = domain_feature.mean(dim=0, keepdim=True)
                _mean_domain_features = mean_domain_feature.repeat_interleave(len(clip_model.labels), dim=0)
                text_features = clip_model._get_text_features(_mean_domain_features.half())

                image_features = clip_model.model.encode_image(image).half()
                # image_features_att = self.model(image_features)
                # image_features = torch.mul(
                    # image_features_att, image_features).detach()
                similarity = clu.get_similarity(image_features, text_features)
                
                _, indices = similarity.topk(1)
                total += len(label)
                pred = torch.squeeze(indices)
                res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
                res = res.cpu().numpy()
                correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])

        return correct/total
    
   


def LongLifeTrain(args, clip_model,appr:Appr, aggNum, from_kbs,idx):
    print('cur round :' + str(aggNum)+'  cur client:' + str(idx))
    taskcla = []
    # acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    # lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    t = aggNum // args.round
    print('cur task:'+ str(t))
    r = aggNum % args.round
    # for t, ncla in taskcla:
    know = False
    if r == args.round - 1:
        know=True
    print('*' * 100)
    # print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    # Get data
    task = t

    # Train
    sws,loss,_ = appr.train( clip_model, task,from_kbs,know)
    print('-' * 100)
    if know:
        return sws,appr.pre_weight['aw'][-1],loss,0
    else:
        return sws, None, loss, 0
