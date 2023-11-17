import torch
import torch.nn as nn
import torch.optim as optim
from model.clip import ClipModelMA, Adapter, Client
import logging
from tqdm import tqdm
from .server import train_server
import math
import numpy as np
from sklearn.cluster import KMeans
from typing import List
from torch.utils.data import TensorDataset, DataLoader
import random


label_parser = lambda x : int(x.split("task")[-1])

def cluster(clients: List[Client], num_experts:int, **kwargs):
    kmeans = KMeans(n_clusters=num_experts)
    flattened_data = [client.preprocess().reshape(-1) for client in clients]
    kmeans.fit(flattened_data)
    cluster_labels = kmeans.labels_
    clustered_data = [[] for _ in range(num_experts)]
    for i, label in enumerate(cluster_labels):
        clustered_data[label].append(i)
        clients[i].assign = label
        temp_list = [0]*num_experts
        temp_list[clients[i].assign] = 1
        clients[i].count_dict = [i/sum(temp_list) for i in temp_list]
    print(clustered_data)
    return clustered_data

@torch.no_grad()    
def randfetch(clip_model: ClipModelMA, dataloader, client:Client,device, **kwargs):
    clip_model.MoE = clip_model.MoE.to(device)
    clip_model.MoE.eval()
    for image, _, _ in dataloader:
        image = image.to(device)
        _ = clip_model.model.encode_image(image).float() # just for the hook work
    return random.randint(0, clip_model.n_experts-1)
         
        
    

@torch.no_grad()
def fetch(clip_model: ClipModelMA, dataloader, client:Client,device, **kwargs):
    clip_model.MoE = clip_model.MoE.to(device)
    clip_model.MoE.eval()
    total = 0
    total_var = 0
    # count_dict = {i:0 for i in range(clip_model.n_experts)}
    count_dict = [0]*clip_model.n_experts
    for image, _, _ in dataloader:
        image = image.to(device)
        _ = clip_model.model.encode_image(image).float() # just for the hook work
    data_feature = torch.tensor(client.preprocess(),dtype=torch.float)
    # print(data_feature[0].shape, type(data_feature[0]))
    dataset = TensorDataset(data_feature)
    feature_dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=True)
    for image_feature in feature_dataloader:
        image_feature = image_feature[0].to(device)
        _, _, logits = clip_model.MoE(image_feature, train_gate=True)
        logits = logits.detach()
        total += len(image)
        total_var += torch.sum(torch.var(logits, axis=1)).item()
        for i in logits:
            count_dict[torch.argmax(i).item()] += 1
    client.count_dict = [i/sum(count_dict) for i in count_dict]
    most_index = max(enumerate(count_dict), key=lambda x:x[1])
    # print(most_index)
    return most_index[0]
    
def communicate(clip_model: ClipModelMA, clients: List[Client], task, device):
    # cluster_data = cluster(clients, clip_model.n_experts)
    cluster_data = [[] for _ in range(clip_model.n_experts)]
    for client in range(len(clients)):
        cluster_data[clients[client].assign].append(client)
    print(cluster_data)
    for index, cluster_clients in enumerate(cluster_data):
        global_adapters = clip_model.MoE.experts[index].to(device)  # FedAvg for every server
        for param in global_adapters.parameters():
            param.data.zero_()
        for c in cluster_clients:
            for global_param, client_param in zip(global_adapters.parameters(), clients[c].adapter.parameters()):
                global_param.data += client_param.data
        for global_param in global_adapters.parameters():
            global_param.data /= len(cluster_clients)  # Average the parameters
        clip_model.MoE.experts[index] = global_adapters
    train_server(clip_model, clients, task, device)
    