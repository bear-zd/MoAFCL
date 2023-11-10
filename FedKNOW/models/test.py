# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/test.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import utils.cliputils as clu
import itertools
from torch.utils.data import TensorDataset

class Client():
    def __init__(self, net, device):
        self.feature_data = []
    def extract_feature(self):
        def hook(module, input, output):
            self.feature_data.append(output.permute(1,0,2).detach().cpu().numpy())
        return hook
    def preprocess(self):
        temp = list(itertools.chain.from_iterable(self.feature_data))
        temp = np.stack(temp, axis = 0)
        temp = np.mean(temp, axis=1)
        return temp
    
from copy import deepcopy
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        d = int(self.idxs[item])
        image, label = self.dataset[d]
        return image, label

class DatasetSplit_leaf(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label
def compute_offsets(task, nc_per_task, is_cifar=True):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2

def test_img_local(clip_model,net_g, test_dataloader,server_data, device=None):
    clip_model.model.eval()
    net_g.to(device)
    net_g.eval()
    correct = 0
    total = 0
    # _,data_loader,_ = dataloader.get_dataloader(t)
    # data_loader = data_loader[user_idx]
    data_feature = torch.tensor(server_data.preprocess(),dtype=torch.float)
    # print(data_feature[0].shape, type(data_feature[0]))
    dataset = TensorDataset(data_feature)
    list_feature_dataloader = list(DataLoader(dataset=dataset, batch_size=50, shuffle=True))
    for index, batch in enumerate(test_dataloader):
        image, _, label = batch
        image, label = image.to(device), label.to(device)
        
        feature = list_feature_dataloader[index][0].to(device)

        image_features = clip_model.model.encode_image(image).float()
        domain_feature =net_g(feature)
        mean_domain_feature = domain_feature.mean(dim=0, keepdim=True)
        _mean_domain_features = mean_domain_feature.repeat_interleave(len(clip_model.labels), dim=0)
        text_features = clip_model._get_text_features(_mean_domain_features.half())
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).float()

        similarity = clu.get_similarity(image_features, text_features)
        _, indices = similarity.topk(1)
       
        total += len(label)
        pred = torch.squeeze(indices)
        res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
        res = res.cpu().numpy()
        correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])
    return  correct,  total



def test_img_local_all_WEIT(clip_model,appr, args, test_dataloader,device=None):
    print('test begin' + '*' * 100)
    server_data = Client(args.net,args.device)
    # variable should be re-implement dict_users_test 
    # net_local = copy.deepcopy(appr[idx].model)
    net_local = copy.deepcopy(appr[0].model)
    net_local.eval()
    all_task_correct = 0
    all_total_num = 0
    for domain_dataloader in test_dataloader:
        # print(torch.cuda.memory_allocated(device))
        temp_hook = clip_model.model.visual.transformer.resblocks[0].register_forward_hook(server_data.extract_feature())
        for image, _, _ in domain_dataloader:
            image = image.to(args.device)
            _ = clip_model.model.encode_image(image).float() # just for the hook work
        temp_hook.remove()
        # print(torch.cuda.memory_allocated(device))
        torch.cuda.empty_cache()
        num_correct, total = test_img_local(clip_model,net_local,domain_dataloader,server_data , device=device)
        all_task_correct += num_correct
        all_total_num += total
        server_data.feature_data = []
        print(f"the domain {domain_dataloader.domain} acc = {num_correct/total}")
        # all_task_acc /= (t + 1)
        # all_task_loss /= (t + 1)
    return all_task_correct, all_total_num

    # if return_all:
    #     return acc_test_local, loss_test_local, all_total_num
    # if write is not None:
    #     write.add_scalar('task_finish_and _agg', sum(acc_test_local) / num_idxxs, t + 1)
    # return sum(acc_test_local) / num_idxxs, sum(loss_test_local) / num_idxxs, num_idxxs


def test_img_local_all_KNOW(clip_model, appr, args, test_dataloaders, t, w_locals=None, w_glob_keys=None, indd=None,
                            dataset_train=None, dict_users_train=None, return_all=False, write=None, num_classes=10,
                            device=None):
    print('test begin' + '*' * 100)
    
    server_data = Client(args.net,args.device)
    net_local = copy.deepcopy(appr[0].model)
    net_local.eval()
    all_task_correct = 0
    all_total_num = 0
    # variable should be re-implement dict_users_test
    for domain_dataloader in test_dataloaders:
        temp_hook = clip_model.model.visual.transformer.resblocks[0].register_forward_hook(server_data.extract_feature())
        for image, _, _ in domain_dataloader:
            image = image.to(args.device)
            _ = clip_model.model.encode_image(image).float() # just for the hook work
        temp_hook.remove()

        torch.cuda.empty_cache()
        num_correct, total = test_img_local(clip_model,net_local,domain_dataloader,server_data , device=device)
        all_task_correct += num_correct
        all_total_num += total
        server_data.feature_data = []
        print(f"the domain {domain_dataloader.domain} acc = {num_correct/total}")
        # all_task_acc /= (t + 1)
        # all_task_loss /= (t + 1)
    return all_task_correct, all_total_num