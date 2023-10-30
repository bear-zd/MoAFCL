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

def test_img_local(clip_model,net_g, dataloader, args,t,idx=None,indd=None, user_idx=-1, idxs=None,appr = None,num_classes=10,device=None):
    clip_model.model.eval()
    net_g.to(device)
    net_g.eval()
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    _,data_loader,_ = dataloader.get_dataloader(t)
    data_loader = data_loader[user_idx]
    texts = clip_model.labels
    text_features = clu.get_text_features_list(texts, clip_model.model, device).float()
    total = 0
    for image, text, label in data_loader:
        image, text, label = image.to(device), text.to(device), label.to(device)
        image_features = clip_model.model.encode_image(image).float()
        # if appr is not None:
        #     image_features_att1 = appr.pernet(image_features)
        #     image_features_att2 = net_g(image_features)
        #     image_features_att = appr.alpha * image_features_att1 + \
        #         (1-appr.alpha)* image_features_att2 
        # else:
        image_features_att = net_g(image_features)
        image_features = torch.mul(image_features_att, image_features).detach()
        ##### this block calculate the loss
        loss_image_features = copy.deepcopy(image_features)
        loss_text_features = clip_model.model.encode_text(text).float().detach()
        loss_image_features = loss_image_features / loss_image_features.norm(dim=1, keepdim=True)
        loss_text_features = loss_text_features / loss_text_features.norm(dim=1, keepdim=True)
        logit_scale = clip_model.model.logit_scale.exp()
        logits_per_image = logit_scale * loss_image_features @ loss_text_features.t()
        logits_per_text = logits_per_image.t()
        ground_truth = torch.arange(len(image), dtype=torch.long, device=device)
        loss = (loss_img(logits_per_image, ground_truth) + 
                loss_txt(logits_per_text, ground_truth))/2
        ##### this block calculate the loss

        image_features = image_features
        similarity = clu.get_similarity(image_features, text_features)

        _, indices = similarity.topk(1)
        total += len(label)
        pred = torch.squeeze(indices)
        res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
        res = res.cpu().numpy()
        correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])
        
        test_loss+=loss.item()

    test_loss /= total
    accuracy = 100.00 * float(correct) / total
    return  accuracy, test_loss, total



def test_img_local_all_WEIT(clip_model,appr, args, dataloader, t, w_locals=None, w_glob_keys=None, indd=None,
                       dataset_train=None, dict_users_train=None, return_all=False, write=None,num_classes = 10,device=None):
    print('test begin' + '*' * 100)
    print('task ' + str(t) + ' finish train')
    # variable should be re-implement dict_users_test 
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    for idx in range(num_idxxs):
        net_local = copy.deepcopy(appr[idx].model)
        net_local.eval()
        all_task_acc = 0
        all_task_loss = 0
        for task in range(t + 1):
            a, b, tot = test_img_local(clip_model,net_local, dataloader, args, task, user_idx=idx, num_classes=num_classes,device=device)
            all_task_acc += a
            all_task_loss += b
        all_task_acc /= (t + 1)
        all_task_loss /= (t + 1)
   
        acc_test_local[idx] = all_task_acc #* tot
        loss_test_local[idx] = all_task_loss #* tot
        del net_local

    if return_all:
        return acc_test_local, loss_test_local
    if write is not None:
        write.add_scalar('task_finish_and _agg', sum(acc_test_local) / tot, t + 1)
    return sum(acc_test_local) / tot, sum(loss_test_local) / tot
