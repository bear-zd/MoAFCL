# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn
import clip
from utils import freeze_param, get_image_features
from model.moe import MoE 
ADAPTER_PARAMETER = {"ViT-B/16":{"image_feature":512, "hidden_size":512, "output_feature":512},
             "RN50":{"image_feature":1024, "hidden_size":512, "output_feature":1024}}

# **PARA_DICT[args.dataset]
class Adapter(nn.Module):
    def __init__(self, base_model: str, label="origin", **kwargs):
        super(Adapter, self).__init__()
        image_feature = ADAPTER_PARAMETER[base_model]['image_feature']
        hidden_size = ADAPTER_PARAMETER[base_model]['hidden_size']
        output_feature = ADAPTER_PARAMETER[base_model]['output_feature']

        self.adapter = nn.Sequential(
                        nn.Linear(image_feature, hidden_size),
                        nn.Tanh(),
                        nn.Linear(hidden_size, output_feature),
                        nn.Softmax(dim=1),
                    )
        self.label = label
    def forward(self, x):
        out = self.adapter(x)
        return out


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


class FedClip(ClipModel):
    def __init__(self,model_name, imgadpy=True, freezepy=True, device="cuda"):
        super().__init__(model_name=model_name, device=device)
        self.imgadpy = imgadpy
        self.freezepy = freezepy 


    def initdgatal(self, dataloader):

        for batch in dataloader:
            image, _, label = batch
            image = image.to(self.device)
            label = label.to(self.device)
            image_features = get_image_features(
                image, self.model, self.preprocess)
            break
        if self.freezepy:
            freeze_param(self.model)

        if self.imgadpy:
            self.img_adap = Adapter(self.model_name).to(self.device)


class ClipModelMA(ClipModel):
    def __init__(self,model_name="Vit-B/32",n_experts=5,top_k = 1, device = "cuda"):
        super().__init__(model_name, device)
        self.n_experts = n_experts
        self.client_label = None
        self.labels = None
        self.top_k = top_k
    
    
    def init_MoE(self):
        freeze_param(self.model)
        init_adapter = Adapter(self.model_name)
        image_feature = ADAPTER_PARAMETER[self.model_name]["image_feature"]
        self.MoE: MoE = MoE(image_feature, init_adapter , num_experts = self.n_experts, k=self.top_k, noisy_gating=True,device=self.device)


class FedWeITClip(ClipModel):
    def __init__(model_name, device):
        super().__init__(model_name, device)
    
    def init_WeIT(self):
        freeze_param(self.model)



