# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from utils.config import img_param_init, set_random_seed

from utils import *
import copy
import argparse
from model.clip import FedClip
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np



def toeval(model):
    model.model.eval()
    model.img_adap.eval()


def totrain(model):
    model.model.train()
    model.img_adap.train()


def train(args, model, data_loader, optimizer, device):
    totrain(model)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    for batch in data_loader:

        image, text, label = batch
        # print(label)
        if len(text) > 1:
            image = image.to(device)
            text = text.to(device)

            image_features = model.model.encode_image(image).float()
            text_features = model.model.encode_text(text).float()
            image_features_att = model.img_adap(image_features)
            image_features = torch.mul(image_features_att, image_features)

            image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)
            text_features = text_features / \
                text_features.norm(dim=1, keepdim=True)

            logit_scale = model.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(
                len(image), dtype=torch.long, device=device)

            loss = (loss_img(logits_per_image, ground_truth) +
                    loss_txt(logits_per_text, ground_truth))/2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test(args, model, data_loader, device):
    toeval(model)
    total = 0
    correct = 0
    texts = model.labels
    text_features = get_text_features_list(texts, model.model, device).float()

    with torch.no_grad():
        for batch in data_loader:

            image, _, label = batch
            image = image.to(device)
            label = label.to(device)
            image_features = get_image_features(
                image, model.model, model.preprocess, device).float()
            image_features_attn = model.img_adap(image_features)
            image_features = torch.mul(
                image_features_attn, image_features).detach()
            similarity = get_similarity(image_features, text_features)
            _, indices = similarity.topk(1)
            total += len(label)
            pred = torch.squeeze(indices)
            res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
            # print(res)
            res = res.cpu().numpy()
            correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])

        return correct, total


def communication(args, server_model, models, client_weights):
    client_num = len(models)
    with torch.no_grad():
        for key in server_model.img_adap.state_dict().keys():
            if 'num_batches_tracked' in key or 'bert' in key:
                server_model.img_adap.state_dict()[key].data.copy_(
                    models[0].img_adap.state_dict()[key])
            else:
                temp = torch.zeros_like(server_model.img_adap.state_dict()[
                                        key], dtype=torch.float32)
                for client_idx in range(client_num):
                    temp += client_weights[client_idx] * \
                        models[client_idx].img_adap.state_dict()[key]
                server_model.img_adap.state_dict()[key].data.copy_(temp)
                for client_idx in range(client_num):
                    models[client_idx].img_adap.state_dict()[key].data.copy_(
                        server_model.img_adap.state_dict()[key])
    return server_model, models


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pacs')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--datapercent', type=float,
                        default=6e-1, help='data percent to use')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--root_dir', type=str, default='../../../data/')
    parser.add_argument('--iters', type=int, default=50,
                        help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=5,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='FedAtImg')
    parser.add_argument('--net', type=str, default='ViT-B/16',
                        help='[RN50 | RN101 | RN50x4 | RN50x16 | RN50x64 | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px]')
    parser.add_argument('--n_task', type=int, default=4, help='task number')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_clients', type=int, default=10)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    args = parser.parse_args()
    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)
    args = img_param_init(args)
    os.makedirs('../data/', exist_ok=True)

    server_model = FedClip(args.net, imgadpy=True, freezepy=True)

    train_loaders, test_loaders, labels = get_data(args.dataset)(args, server_model.preprocess).get_dataloader(0)
    server_model.labels = labels
    

    server_model.initdgatal(test_loaders[0])

    client_num = args.n_clients
    client_weights = [1/client_num for i in range(client_num)]
    models = [copy.deepcopy(server_model)for idx in range(client_num)]
    for i in range(client_num):
        models[i].labels = labels
        models[i].model.to(device)
        models[i].img_adap.to(device)
    best_changed = False

    best_acc = [0. for j in range(client_num)]
    for a_iter in range(args.n_task):
        optimizers = [optim.Adam(params=[{'params': models[idx].img_adap.parameters()}], lr=args.lr, betas=(
            args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay) for idx in range(client_num)]
        train_loaders, test_loaders, labels = get_data(args.dataset)(args, server_model.preprocess).get_dataloader(a_iter)    
        for client_idx, model in enumerate(models):
            print(f"=========  Trian client {client_idx} ===========")
            for wi in range(args.wk_iters):
                # print("    == Train iters {} task {} ============".format(wi + a_iter * args.wk_iters,a_iter))
                train(
                    args, model, train_loaders[client_idx], optimizers[client_idx], device)
                if wi%5==0:
                    train_acc, total = test(args, model, train_loaders[client_idx], device)
                    print(' Site-{}| Train Acc: {}'.format(client_idx, train_acc/total))
        with torch.no_grad():
            server_model, models = communication(
                args, server_model, models, client_weights)

            # val_acc_list = [0. for j in range(client_num)]
            # do clients test
            # for client_idx, model in enumerate(models):
                
            # for domain_idx in range(len(args.domains)):
            #     val_acc, total = test(
            #         args, server_model, test_loaders[domain_idx], device)
            #     val_acc_list[client_idx] = val_acc
            #     print(' Site-{}| Val  Acc: {}'.format(
            #         client_idx, val_acc/ total), flush=True)


            t, c = 0, 0
            for domain_idx in range(len(test_loaders)):
                correct, total= test(args, server_model,
                                    test_loaders[domain_idx], device)
                t += total
                c += correct
                print(
                    ' Test site-{}| Test Acc: {}'.format(test_loaders[domain_idx].domain, correct/total))
            print(f"total acc:{c/t}")


