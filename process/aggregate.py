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
    print(clustered_data)
    return clustered_data
    

         
        
    

@torch.no_grad()
def fetch(clip_model: ClipModelMA, dataloader, client:Client,device, **kwargs):
    clip_model.MoE = clip_model.MoE.to(device)
    clip_model.MoE.eval()
    total = 0
    total_var = 0
    count_dict = {i:0 for i in range(clip_model.n_experts)}
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
    most_index = max(count_dict, key=count_dict.get)
    # print(most_index)
    return most_index
    
def communicate(clip_model: ClipModelMA, clients: List[Client], device):
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
    train_server(clip_model, clients, device)
    
    

        

    

# def communicate(clip_model: ClipModelMA, new_adapter: Adapter, dataloader, device):
#     model_client_labels = [i.label for i in clip_model.MoE.experts] # get label from the expert model
#     if "origin" in model_client_labels: # replace the untrained original expert
#         for index, label in enumerate(model_client_labels):
#             if label == "origin":
#                 del(clip_model.MoE.experts[index])
#                 clip_model.MoE.experts.insert(index, new_adapter)
#                 replace_client_index = index
#                 break
#     else:
#         least_client = math.inf
#         replace_client_index = 0
#         print(model_client_labels)
#         model_client_labels = [label_parser(i) for i in model_client_labels] # replace the most old model
        
#         for index, label in enumerate(model_client_labels):
#             replace_client_index = index if label < least_client else replace_client_index
#             least_client = min(least_client, label)
#         del(clip_model.MoE.experts[replace_client_index])
#         clip_model.MoE.experts.insert(replace_client_index, new_adapter)
#     train_server(clip_model, dataloader, replace_client_index, device)
        



    