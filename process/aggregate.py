import torch
import torch.nn as nn
import torch.optim as optim
from model.clip import ClipModelMA, Adapter
import logging
from tqdm import tqdm
from .server import train_server
import math

label_parser = lambda x : int(x.split("task")[-1])



def communicate(clip_model: ClipModelMA, new_adapter, dataloader, device):
    model_client_labels = [i.label for i in clip_model.MoE.experts] # get label from the expert model
    if "origin" in model_client_labels: # replace the untrained original expert
        for index, label in enumerate(model_client_labels):
            if label == "origin":
                del(clip_model.MoE.experts[index])
                clip_model.MoE.experts.insert(index, new_adapter)
                replace_client_index = index
                break
    else:
        least_client = math.inf
        replace_client_index = 0
        print(model_client_labels)
        model_client_labels = [label_parser(i) for i in model_client_labels] # replace the most old model
        
        for index, label in enumerate(model_client_labels):
            replace_client_index = index if label < least_client else replace_client_index
            least_client = min(least_client, label)
        del(clip_model.MoE.experts[replace_client_index])
        clip_model.MoE.experts.insert(replace_client_index, new_adapter)
    train_server(clip_model, dataloader, replace_client_index, device)
        



    