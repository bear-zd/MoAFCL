import sys
import os
import copy
from argument import argparser
from typing import List
import time

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch.optim as optim
import numpy as np
from tqdm import tqdm
import logging

from utils.config import img_param_init, set_random_seed
from utils.dataload import get_data
from model.clip import ClipModelMA, Adapter
from process import train_client, communicate, test_server, test_client


def init():
    args = argparser()
    if args.logdir is not None:
        logging.basicConfig(filename=args.logdir, level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)
    args = img_param_init(args)  # init the dataset parameters(domains and classnum)
    return args


def main():
    args = init()
    logging.info("Argument init successful!")
    server_model: ClipModelMA = ClipModelMA(args.net, n_experts=args.n_experts, device=args.device)  # load the server data
    train_loaders, test_loaders, labels = get_data(args.dataset)(args, server_model.preprocess).get_dataloader()
    assert len(train_loaders) == args.n_clients, "This just mention you to make sure the n_clients \
                                                  equals to the datasplit client number."
    server_model.labels = labels
    logging.info("Data init successful!")
    
    server_model.init_MoE() # expert number already saved in the server_model.
    logging.info("Model MoE init successful!")  # use the structure of single CLIP training server and client

    for i in range(args.n_clients):
        logging.info(f"client {i} start to train!")
        
        image_adapter = Adapter(args.net)   
        
        optimizer = optim.Adam(params=image_adapter.parameters(), lr=args.lr, betas=(
            args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay) 
        server_model.model.to(args.device)
        image_adapter.to(args.device)
        for epoch in tqdm(range(args.inner_iter)):
            train_client(server_model, image_adapter, train_loaders[i], optimizer ,args.device)
            train_acc = test_client(server_model, image_adapter, train_loaders[i],  args.device)
            test_acc = test_client(server_model, image_adapter, test_loaders[i],  args.device)

        logging.info(f"client {i} - inner_iter: {epoch}, test_acc: {test_acc}, train_acc: {train_acc}")
            # there should be a chose from user data, but not now
        logging.info(f"client {i} start to communication!")
        communicate(server_model, image_adapter, train_loaders[i], args.device)
        logging.info(f"server adapters:{[i.label for i in server_model.MoE.experts]}")
        logging.info(f"server start to eval!")
        test_acc_list, train_acc_list = [], []
        for j in range(i+1):
            test_acc_list.append(test_server(server_model, test_loaders[j], args.device))
            train_acc_list.append(test_server(server_model, train_loaders[j], args.device))
        logging.info(f"the {i} turn eval result: test_acc {test_acc_list}\t train_acc {train_acc_list}")
       
       
if __name__ == "__main__":
    main()