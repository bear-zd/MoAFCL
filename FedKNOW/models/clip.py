# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn
import clip
from utils.cliputils import freeze_param
import copy
import torch
from .layer import DecomposedLinear
ADAPTER_PARAMETER = {"ViT-B/32":{"image_feature":512, "hidden_size":1024, "output_feature":512, "extract_feature":768},
                    "ViT-B/16":{"image_feature":512, "hidden_size":1024, "output_feature":512, "extract_feature":768},
                     "ViT-L/14":{"image_feature":768, "hidden_size":1024, "output_feature":512, "extract_feature":1024},
                     "ViT-L/14@336px":{"image_feature":768, "hidden_size":1024, "output_feature":512, "extract_feature":1024}}

    
class WeITAdapter(nn.Module):
    def __init__(self, base_model: str, label="origin", **kwargs):
        super().__init__()
        input_size = ADAPTER_PARAMETER[base_model]['extract_feature']
        hidden_size = ADAPTER_PARAMETER[base_model]['hidden_size']
        output_size = ADAPTER_PARAMETER[base_model]['image_feature']
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
        self.labels = None
        self.EMBEDDING_DIM = ADAPTER_PARAMETER[self.model_name]["image_feature"]



class FedWeITClip(ClipModel):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
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
    

class FedKNOWClip(FedWeITClip):
    pass



