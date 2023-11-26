# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn
import clip
from utils import freeze_param, get_image_features
from model.moe import MoE 
import itertools
import numpy as np
import torch
import torch.nn.functional as F
ADAPTER_PARAMETER = {"ViT-B/16":{"image_feature":512, "hidden_size":1024, "output_feature":512, "extract_feature":768},
             "RN50":{"image_feature":1024, "hidden_size":1024, "output_feature":1024,"extract_feature":512}}

class Client():
    def __init__(self, net, device):
        self.feature_data = []
        self.adapter: VisionDomainAdapter = VisionDomainAdapter(net, 8) 
        self.assign = None
        self.count_dict = None
    def extract_feature(self):
        def hook(module, input, output):
            self.feature_data.append(output.permute(1,0,2).detach().cpu().numpy())
        return hook
    def preprocess(self):
        temp = list(itertools.chain.from_iterable(self.feature_data))
        temp = np.stack(temp, axis = 0)
        temp = np.mean(temp, axis=1)
        return temp

class VisionDomainAdapter(nn.Module):
    def __init__(self, base_model, domain_token):
        super(VisionDomainAdapter, self).__init__()
        image_feature = ADAPTER_PARAMETER[base_model]['extract_feature']
        hidden_size = ADAPTER_PARAMETER[base_model]['hidden_size']
        output_feature = ADAPTER_PARAMETER[base_model]['output_feature']
        self.input = nn.Linear(image_feature, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(hidden_size, 512*domain_token)
        self.n_outputs = output_feature
    
    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.output(x)
        return x


    
    

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
        self.EMBEDDING_DIM = 512


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
    def __init__(self,model_name="Vit-B/32",n_adapters=5,top_k = 1, device = "cuda"):
        super().__init__(model_name, device)
        self.n_adapters = n_adapters
        self.client_label = None
        self.labels = None
        self.top_k = top_k
        self.client_feature = {}
        self.client_adapter = []
        self.model_name = model_name
        # self.EMBEDDING_DIM = 512
        freeze_param(self.model)
    
    def init_prompt(self, domain_token = 8, sentence_prompt = False):
        self.domain_token = domain_token
        prompt_prefix = ' '.join(['X'] * domain_token )
        if sentence_prompt:
            print('Using sentence_prompt in DPLCLIP...')
            classnames = [f"a photo of a {name.replace('_', ' ')}" for name in len(self.labels)]
        else:
            classnames = [name.replace('_', ' ') for name in self.labels]
        prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]

        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)

        with torch.no_grad():
            embedding = self.model.token_embedding(self.tokenized_prompts).type(self.model.dtype)
        
        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, domain_token + 1:, :]  # CLS, EOS
    
    def _get_text_features(self, domain_feature, coop=False):
        #  reshape domain_feature: [7, 16 * self.EMBEDDING_DIM] -> [7, 16, self.EMBEDDING_DIM]
        domain_feature = domain_feature.reshape(-1, self.domain_token, self.EMBEDDING_DIM)
        
        #  reshape domain_feature: [7, 16, self.EMBEDDING_DIM] -> [7, 77, self.EMBEDDING_DIM]
        domain_feature = torch.cat([self.token_prefix, domain_feature, self.token_suffix], dim=1)
        
        #  refer CoOp: CoOP github. https://github.com/KaiyangZhou/CoOp/blob/b0a058869cef00a4e4ea5256d40fd7681119c099/trainers/coop.py#L46
        x = domain_feature + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x).type(self.model.dtype)
        
        #  mapping domain_features to text_features.
        text_features = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.model.text_projection      
        return text_features
    
    def init_MoE(self):
        # init_adapter = Adapter(self.model_name)
        init_adapter = VisionDomainAdapter(self.model_name, self.domain_token)
        image_feature = ADAPTER_PARAMETER[self.model_name]["image_feature"]
        extract_feature = ADAPTER_PARAMETER[self.model_name]["extract_feature"]
        self.MoE: MoE = MoE(extract_feature ,image_feature, init_adapter , num_adapters = self.n_adapters, k=self.top_k, noisy_gating=True,device=self.device)


class FedWeITClip(ClipModel):
    def __init__(self,model_name, device):
        super().__init__(model_name, device)
        freeze_param(self.model)
    
    def init_WeIT(self):
        freeze_param(self.model)

class FedKNOWClip(ClipModel):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        freeze_param(self.model)
        


