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
from model.layer import DecomposedLinear

ADAPTER_PARAMETER = {"ViT-B/32":{"image_feature":512, "hidden_size":1024, "output_feature":512, "extract_feature":768},
                    "ViT-B/16":{"image_feature":512, "hidden_size":1024, "output_feature":512, "extract_feature":768},
                     "ViT-L/14":{"image_feature":768, "hidden_size":1024, "output_feature":512, "extract_feature":1024},
                     "ViT-L/14@336px":{"image_feature":768, "hidden_size":1024, "output_feature":512, "extract_feature":1024}}

class KNOWAdapter(nn.Module):
    def __init__(self, base_model : str, device=None, **kwargs):
        super().__init__()
        input_size = ADAPTER_PARAMETER[base_model]['extract_feature']
        hidden_size = ADAPTER_PARAMETER[base_model]['hidden_size']
        output_size = ADAPTER_PARAMETER[base_model]['image_feature']
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
        output_feature = ADAPTER_PARAMETER[base_model]['image_feature']
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

