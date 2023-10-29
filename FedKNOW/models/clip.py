# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn
import clip
from utils.cliputils import freeze_param
import copy
from model.moe import MoE 
from .layer import DecomposedLinear
ADAPTER_PARAMETER = {"ViT-B/16":{"image_feature":512, "hidden_size":512, "output_feature":512},
             "RN50":{"image_feature":1024, "hidden_size":512, "output_feature":1024}}


    
class WeITAdapter(nn.Module):
    def __init__(self, base_model: str, label="origin", **kwargs):
        super().__init__()
        input_size = ADAPTER_PARAMETER[base_model]['image_feature']
        output_size = ADAPTER_PARAMETER[base_model]['output_feature']
        hidden_size = ADAPTER_PARAMETER[base_model]['hidden_size']
        self.fc1 = DecomposedLinear(input_size, hidden_size)
        self.last = DecomposedLinear(hidden_size, output_size)

    def set_sw(self,glob_weights):
        self.fc1.sw = glob_weights[0]
    
    def set_knowledge(self, t, from_kbs):
        self.fc1.set_atten(t, from_kbs[0].size(-1))
        self.fc1.set_knlwledge(from_kbs[0])
    
    def get_weights(self):
        weights = []
        w = self.fc1.get_weight().detach()
        w.requires_grad = False
        weights.append(w)
    def forward(self, x, t):
        m = nn.Tanh(self.fc1(x))
        output = nn.Softmax(self.last(m))
        return output
    


class ClipModel(object):
    def __init__(self, model_name: str, device="cuda") -> None:
        CLIP_MODELS = [
        "RN50",
        "RN101",
        "RN50x4",
        "RN50x16",
        "RN50x64",
        "ViT-B/32",
        "ViT-B/16",
        "ViT-L/14",
        "ViT-L/14@336px",
        ]
        if model_name not in CLIP_MODELS:
            raise Exception(f"model name {model_name} not included")
        self.model, self.preprocess = clip.load(model_name, device=device, jit=False)
        self.model.eval()
        self.model.to(device)
        self.model_name = model_name
        self.device = device



class FedWeITClip(ClipModel):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        freeze_param(self.model)
        



