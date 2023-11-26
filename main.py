import sys
import os
from argument import argparser
import itertools
import copy

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch.optim as optim
import numpy as np
from tqdm import tqdm
import logging

from utils.config import img_param_init, set_random_seed
from utils.dataload import DomainDataset, get_data
from model.clip import ClipModelMA, Adapter, Client
from process import train_client, communicate, test_server, test_client, fetch, cluster, randfetch


def init():
    args = argparser()
    if args.logdir is not None:
        logging.basicConfig(filename=args.logdir, level=logging.INFO)
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)
    args = img_param_init(args)  # init the dataset parameters(domains and classnum)
    args.thresh = 1e-4
    return args


def main():
    args = init()
    print(args.rand, type(args.rand))
    logging.info("Argument init successful!")
    server_model: ClipModelMA = ClipModelMA(args.net, n_adapters=args.n_adapters, device=args.device)  # load the server data
    clients_data  = [copy.deepcopy(Client(args.net,args.device)) for i in range(args.n_clients)] # load the client data
    dataloader: DomainDataset = get_data(args.dataset)(args, server_model.preprocess)
    train_loaders, test_loaders, labels = dataloader.get_dataloader(0)
    assert len(train_loaders) == args.n_clients, "This just mention you to make sure the n_clients \
                                                equals to the datasplit client number."
    server_model.labels = labels
    logging.info("Data init successful!")
    # print(len(test_loaders))
    server_model.init_prompt()
    server_model.init_MoE() # adapter number already saved in the server_model.
    
    logging.info("Model MoE init successful!")  # use the structure of single CLIP training server and client
    
    for task in range(args.n_task):
        logging.info(f"task{task} start train!")
        train_loaders, test_loaders, labels = dataloader.get_dataloader(task)
        if (task == 0): # start up
            logging.info("first round start cluster!")
            
            for client in tqdm(range(args.n_clients)):
                clients_data[client].feature_data = []
                clients_data[client].temp_hook = server_model.model.visual.transformer.resblocks[args.extract_layer].register_forward_hook(clients_data[client].extract_feature())
                acc_list = []
                for i in range(args.inner_iter):
                    train_client(server_model, clients_data[client], train_loaders[client], args.device, args)
                    acc_list.append(test_client(server_model,  clients_data[client], train_loaders[client], args.device))
                # no_adapter_acc = test_client(server_model,  None, train_loaders[client], args.device)
                logging.info(f'task {task} - client {client} - acc_list{acc_list}')
            cluster(clients_data, server_model.n_adapters)
            
        else:
            if args.rand == 1:
                logging.info(f"{task} round start fetch!")
            else:
                logging.info(f"{task} round start randfetch!")
            for client in tqdm(range(args.n_clients)):
                clients_data[client].feature_data = []
                temp_hook = server_model.model.visual.transformer.resblocks[args.extract_layer].register_forward_hook(clients_data[client].extract_feature())
                if args.rand == 1:
                    clients_data[client].assign = randfetch(server_model, train_loaders[client], clients_data[client],args.device)
                else:   
                    clients_data[client].assign = fetch(server_model, train_loaders[client], clients_data[client],args.device)
                temp_hook.remove()
                acc_list = []
                # clients_data[client].feature_data = []
                clients_data[client].adapter = copy.deepcopy((server_model.MoE.adapters[clients_data[client].assign]))
                # temp_hook = server_model.model.visual.transformer.resblocks[0].register_forward_hook(clients_data[client].extract_feature())
                for i in range(args.inner_iter):
                    train_client(server_model, clients_data[client], train_loaders[client], args.device, args)
                    acc_list.append(test_client(server_model,  clients_data[client], train_loaders[client], args.device))
                logging.info(f'task {task} - client {client} - acc_list{acc_list}')
                # temp_hook.remove()

        communicate(server_model, clients_data, task, args.device)
        for client in range(args.n_clients):
            clients_data[client].feature_data = []
        server_data = Client(args.net,args.device)
        total, correct = 0, 0
        # print(len(test_loaders))
        for i in test_loaders:
            temp_hook = server_model.model.visual.transformer.resblocks[args.extract_layer].register_forward_hook(server_data.extract_feature())
            for image, _, _ in i:
                image = image.to(args.device)
                _ = server_model.model.encode_image(image).float() # just for the hook work
            temp_hook.remove()
            len_t, correct_t = test_server(server_model, i,  server_data,args.device)
            total += len_t; correct += correct_t
            server_data.feature_data = []
            logging.info(f"task {task} - {i.domain} test_acc: {correct_t/len_t}")
        logging.info(f"task {task} total test_acc: {correct/total}")

            
            
       
if __name__ == "__main__":
    main()