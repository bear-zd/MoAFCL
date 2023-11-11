# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/Nets.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision import models
import itertools

# from models.ResNetWEIT import resnetWEIT18
import json
import numpy as np

# from models.layer import DecomposedConv,DecomposedLinear
from models.layer import DecomposedLinear

ADAPTER_PARAMETER = {"ViT-B/16":{"image_feature":512, "hidden_size":1024, "output_feature":512, "extract_feature":768},
             "RN50":{"image_feature":1024, "hidden_size":1024, "output_feature":1024,"extract_feature":512}}


class KNOWAdapter(nn.Module):
    def __init__(self, base_model : str, device=None, **kwargs):
        super().__init__()
        input_size = ADAPTER_PARAMETER[base_model]['extract_feature']
        output_size = ADAPTER_PARAMETER[base_model]['output_feature']
        hidden_size = ADAPTER_PARAMETER[base_model]['hidden_size']
        self.feature_net = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.last = nn.Linear(hidden_size, 8*output_size)
        self.softmax = nn.Softmax(1)
        self.weight_keys = [['feature_net.fc1.weight'], ['feature_net.fc1.bias'], ['last.weight'], ['last.bias']]

    def forward(self, x):
        m = self.feature_net(x)
        m = self.dropout(m)
        m = self.last(m)
        output = self.softmax(m)
        return output

class Client():
    def __init__(self, net, device):
        self.feature_data = []
        self.adapter: WeITAdapter = WeITAdapter(net, device) 
        self.assign = None
    def extract_feature(self):
        def hook(module, input, output):
            self.feature_data.append(output.permute(1,0,2).detach().cpu().numpy())
        return hook
    def preprocess(self):
        temp = list(itertools.chain.from_iterable(self.feature_data))
        temp = np.stack(temp, axis = 0)
        temp = np.mean(temp, axis=1)
        return temp


    
class WeITAdapter(nn.Module):
    def __init__(self, base_model: str,  device=None, **kwargs):
        super().__init__()
        image_feature = ADAPTER_PARAMETER[base_model]['extract_feature']
        hidden_size = ADAPTER_PARAMETER[base_model]['hidden_size']
        output_feature = ADAPTER_PARAMETER[base_model]['output_feature']
        # print(device)
        self.weight_keys = []
        self.fc1 = DecomposedLinear(image_feature, hidden_size, device=device)
        self.dropout = nn.Dropout(0.1)
        self.last = DecomposedLinear(hidden_size, output_feature*8 ,device=device)
        self.softmax = nn.Softmax(1)

    def set_sw(self,glob_weights):
        self.fc1.sw = Parameter(glob_weights[0])
        self.last.sw = Parameter(glob_weights[1])
    
    def set_knowledge(self, t, from_kbs):
        self.fc1.set_atten(t, from_kbs[0].size(-1))
        self.fc1.set_knlwledge(from_kbs[0])
        self.last.set_atten(t, from_kbs[1].size(-1))
        self.last.set_knlwledge(from_kbs[1])

    def get_weights(self):
        weights = []
        w = self.fc1.get_weight().detach()
        w.requires_grad = False
        weights.append(w)
        w = self.last.get_weight().detach()
        w.requires_grad = False
        weights.append(w)
        return weights
    def forward(self, x):
        m = self.dropout(self.fc1(x))
        m = F.relu(m)
        output = self.softmax(self.last(m))
        return output

        # h = self.fc2(h)
# class WEITResNet(nn.Module):
#     def __init__(self,output=100,nc_per_task = 10):
#         super().__init__()

#         self.feature_net = resnetWEIT18()
#         self.last = DecomposedLinear(self.feature_net.outlen, output)
#         self.weight_keys = []
#         for name,para in self.named_parameters():
#             temp=[]
#             if 'fc' not in name:
#                 temp.append(name)
#                 self.weight_keys.append(temp)
#     def set_sw(self,glob_weights):
#         self.feature_net.conv1.sw = Parameter(glob_weights[0])
#         self.feature_net.layer1[0].conv1.sw = Parameter(glob_weights[1])
#         self.feature_net.layer1[0].conv2.sw = Parameter(glob_weights[2])
#         self.feature_net.layer1[1].conv1.sw = Parameter(glob_weights[3])
#         self.feature_net.layer1[1].conv2.sw = Parameter(glob_weights[4])
#         self.feature_net.layer2[0].conv1.sw = Parameter(glob_weights[5])
#         self.feature_net.layer2[0].conv2.sw = Parameter(glob_weights[6])
#         self.feature_net.layer2[1].conv1.sw = Parameter(glob_weights[7])
#         self.feature_net.layer2[1].conv2.sw = Parameter(glob_weights[8])
#         self.feature_net.layer3[0].conv1.sw = Parameter(glob_weights[9])
#         self.feature_net.layer3[0].conv2.sw = Parameter(glob_weights[10])
#         self.feature_net.layer3[1].conv1.sw = Parameter(glob_weights[11])
#         self.feature_net.layer3[1].conv2.sw = Parameter(glob_weights[12])
#         self.feature_net.layer4[0].conv1.sw = Parameter(glob_weights[13])
#         self.feature_net.layer4[0].conv2.sw = Parameter(glob_weights[14])
#         self.feature_net.layer4[1].conv1.sw = Parameter(glob_weights[15])
#         self.feature_net.layer4[1].conv2.sw = Parameter(glob_weights[16])
#         self.last.sw = glob_weights[17]
#     def set_knowledge(self,t,from_kbs):
#         self.feature_net.conv1.set_atten(t,from_kbs[0].size(-1))
#         self.feature_net.conv1.set_knlwledge(from_kbs[0])
#         self.feature_net.layer1[0].conv1.set_atten(t, from_kbs[1].size(-1))
#         self.feature_net.layer1[0].conv1.set_knlwledge(from_kbs[1])
#         self.feature_net.layer1[0].conv2.set_atten(t, from_kbs[2].size(-1))
#         self.feature_net.layer1[0].conv2.set_knlwledge(from_kbs[2])
#         self.feature_net.layer1[1].conv1.set_atten(t, from_kbs[3].size(-1))
#         self.feature_net.layer1[1].conv1.set_knlwledge(from_kbs[3])
#         self.feature_net.layer1[1].conv2.set_atten(t, from_kbs[4].size(-1))
#         self.feature_net.layer1[1].conv2.set_knlwledge(from_kbs[4])

#         self.feature_net.layer2[0].conv1.set_atten(t, from_kbs[5].size(-1))
#         self.feature_net.layer2[0].conv1.set_knlwledge(from_kbs[5])
#         self.feature_net.layer2[0].conv2.set_atten(t, from_kbs[6].size(-1))
#         self.feature_net.layer2[0].conv2.set_knlwledge(from_kbs[6])
#         self.feature_net.layer2[1].conv1.set_atten(t, from_kbs[7].size(-1))
#         self.feature_net.layer2[1].conv1.set_knlwledge(from_kbs[7])
#         self.feature_net.layer2[1].conv2.set_atten(t, from_kbs[8].size(-1))
#         self.feature_net.layer2[1].conv2.set_knlwledge(from_kbs[8])

#         self.feature_net.layer3[0].conv1.set_atten(t, from_kbs[9].size(-1))
#         self.feature_net.layer3[0].conv1.set_knlwledge(from_kbs[9])
#         self.feature_net.layer3[0].conv2.set_atten(t, from_kbs[10].size(-1))
#         self.feature_net.layer3[0].conv2.set_knlwledge(from_kbs[10])
#         self.feature_net.layer3[1].conv1.set_atten(t, from_kbs[11].size(-1))
#         self.feature_net.layer3[1].conv1.set_knlwledge(from_kbs[11])
#         self.feature_net.layer3[1].conv2.set_atten(t, from_kbs[12].size(-1))
#         self.feature_net.layer3[1].conv2.set_knlwledge(from_kbs[12])

#         self.feature_net.layer4[0].conv1.set_atten(t, from_kbs[13].size(-1))
#         self.feature_net.layer4[0].conv1.set_knlwledge(from_kbs[13])
#         self.feature_net.layer4[0].conv2.set_atten(t, from_kbs[14].size(-1))
#         self.feature_net.layer4[0].conv2.set_knlwledge(from_kbs[14])
#         self.feature_net.layer4[1].conv1.set_atten(t, from_kbs[15].size(-1))
#         self.feature_net.layer4[1].conv1.set_knlwledge(from_kbs[15])
#         self.feature_net.layer4[1].conv2.set_atten(t, from_kbs[16].size(-1))
#         self.feature_net.layer4[1].conv2.set_knlwledge(from_kbs[16])

#         self.last.set_atten(t, from_kbs[17].size(-1))
#         self.last.set_knlwledge(from_kbs[17])
#     def get_weights(self):
#         weights = []

#         w = self.feature_net.conv1.get_weight().detach()
#         w.requires_grad = False
#         weights.append(w)

#         w = self.feature_net.layer1[0].conv1.get_weight().detach()
#         weights.append(w)
#         w.requires_grad = False
#         w = self.feature_net.layer1[0].conv2.get_weight().detach()
#         weights.append(w)
#         w.requires_grad = False
#         w = self.feature_net.layer1[1].conv1.get_weight().detach()
#         weights.append(w)
#         w.requires_grad = False
#         w = self.feature_net.layer1[1].conv2.get_weight().detach()
#         weights.append(w)
#         w.requires_grad = False

#         w = self.feature_net.layer2[0].conv1.get_weight().detach()
#         weights.append(w)
#         w.requires_grad = False
#         w = self.feature_net.layer2[0].conv2.get_weight().detach()
#         weights.append(w)
#         w.requires_grad = False
#         w = self.feature_net.layer2[1].conv1.get_weight().detach()
#         weights.append(w)
#         w.requires_grad = False
#         w = self.feature_net.layer2[1].conv2.get_weight().detach()
#         weights.append(w)
#         w.requires_grad = False

#         w = self.feature_net.layer3[0].conv1.get_weight().detach()
#         weights.append(w)
#         w.requires_grad = False
#         w = self.feature_net.layer3[0].conv2.get_weight().detach()
#         weights.append(w)
#         w.requires_grad = False
#         w = self.feature_net.layer3[1].conv1.get_weight().detach()
#         weights.append(w)
#         w.requires_grad = False
#         w = self.feature_net.layer3[1].conv2.get_weight().detach()
#         weights.append(w)
#         w.requires_grad = False

#         w = self.feature_net.layer4[0].conv1.get_weight().detach()
#         weights.append(w)
#         w.requires_grad = False
#         w = self.feature_net.layer4[0].conv2.get_weight().detach()
#         weights.append(w)
#         w.requires_grad = False
#         w = self.feature_net.layer4[1].conv1.get_weight().detach()
#         weights.append(w)
#         w.requires_grad = False
#         w = self.feature_net.layer4[1].conv2.get_weight().detach()
#         weights.append(w)
#         w.requires_grad = False

#         w = self.last.get_weight().detach()
#         weights.append(w)
#         w.requires_grad = False

#         return weights
#     def forward(self,x,t,pre=False,is_con=False):
#         h = self.feature_net(x)
#         output = self.last(h)
#         if is_con:
#             # make sure we predict classes within the current task
#             if pre:
#                 offset1 = 0
#                 offset2 = int(t  * self.nc_per_task)
#             else:
#                 offset1 = int(t * self.nc_per_task)
#                 offset2 = int((t + 1) * self.nc_per_task)
#             if offset1 > 0:
#                 output[:, :offset1].data.fill_(-10e10)
#             if offset2 < self.n_outputs:
#                 output[:, offset2:self.n_outputs].data.fill_(-10e10)
#         return output

# class Classification(nn.Module):
#     def __init__(self, output=100, nc_per_task=10):
#         super().__init__()

#         self.last = nn.Linear(1024, output)
#         self.nc_per_task=nc_per_task
#         self.n_outputs = output

#     def forward(self, h, t, pre=False, is_cifar=True):
#         output = self.last(h)
#         if is_cifar:
#             # make sure we predict classes within the current task
#             if pre:
#                 offset1 = 0
#                 offset2 = int(t * self.nc_per_task)
#             else:
#                 offset1 = int(t * self.nc_per_task)
#                 offset2 = int((t + 1) * self.nc_per_task)
#             if offset1 > 0:
#                 output[:, :offset1].data.fill_(-10e10)
#             if offset2 < self.n_outputs:
#                 output[:, offset2:self.n_outputs].data.fill_(-10e10)
#         return output
