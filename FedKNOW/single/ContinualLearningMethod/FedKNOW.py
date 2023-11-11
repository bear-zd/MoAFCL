import sys, time, os
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from utils import *
from torch.utils.tensorboard import SummaryWriter
import quadprog
sys.path.append('..')
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
import itertools
from torch.utils.data import TensorDataset, DataLoader
import utils.cliputils as clu



def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1
def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1
def project2cone2(gradient, memories,memory, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    try:
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        gradient.copy_(torch.Tensor(x).view(-1, 1))
    except ValueError:
        memory_np = memory.cpu().t().double().numpy()
        t = memory_np.shape[0]
        P = np.dot(memory_np, memory_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
        q = np.dot(memory_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + margin
        try:
            v = quadprog.solve_qp(P, q, G, h)[0]
            x = np.dot(v, memory_np) + gradient_np
            gradient.copy_(torch.Tensor(x).view(-1, 1))
        except ValueError:
            gradient.copy_(torch.Tensor(gradient_np).view(-1, 1))
def MultiClassCrossEntropy(logits, labels, t=None,T=2):
    # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    outputs = torch.log_softmax(logits / T, dim=1)  # compute the log of softmax values
    label = torch.softmax(labels / T, dim=1)
        # print('outputs: ', outputs)
        # print('labels: ', labels.shape)
    outputs = torch.sum(outputs * label, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)

    # print('OUT: ', outputs)
    return outputs

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return
class Appr(object):
    def __init__(self, model, packnet,packmodel, tr_dataloader,nepochs=100, lr=0.001, lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=100,
                 args=None):
        self.device = args.device
        self.num_classes = args.num_classes
        self.model = model
        self.model_old = model
        self.pack = packnet
        self.packmodel = packmodel
        self.fisher = None
        self.nepochs = nepochs
        self.tr_dataloader = tr_dataloader
        self.lr = lr
        self.lr_decay = args.lr_decay
        self.lr_min = lr_min * 1 / 3
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.ce = torch.nn.CrossEntropyLoss()
        self.optim_type = args.optim
        self.optimizer = self._get_optimizer()
        self.pack_optimizer = self._get_optimizer(packmodel)
        self.lamb = args.lamb
        # self.e_rep = args.local_local_ep
        self.e_rep = 1
        self.old_task=-1
        self.grad_dims = []
        self.num_classes = args.num_classes # 不确定
        self.pre_weight = {
            'weight':[],
            'aw':[],
            'mask':[]
        }
        for param in self.model.feature_net.parameters():
            self.grad_dims.append(param.data.numel())
        self.select_grad_num = args.select_grad_num

        self.device = args.device
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

    def set_sw(self, glob_weights):
        i = 0
        keys = [k for k, _ in self.model.named_parameters()]
        if len(glob_weights) > 0:
            all_weights = []
            for name, para in self.model.named_parameters():
                if 'sw' in name:
                    all_weights.append(glob_weights[i])
                    i = i + 1
                else:
                    all_weights.append(para)
            model_dict = self.model.state_dict()
            feature_dict = zip(keys, all_weights)
            # last_dict = OrderedDict({k: torch.Tensor(v) for k, v in zip(last_keys,last_para)})
            save_model = OrderedDict({k: v for k, v in feature_dict})
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)

    def set_model(self,model):
        self.model = model
    def set_fisher(self,fisher):
        self.fisher = fisher
    def set_trData(self,tr_dataloader):
        self.tr_dataloader = tr_dataloader
    def _get_optimizer(self, model=None,lr=None):
        if lr is None: lr = self.lr
        optimizer =None
        if model == None:
            if "SGD" in self.optim_type:
                optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.lr_decay)
            else:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,weight_decay=self.lr_decay)
        else:
            if "SGD" in self.optim_type:
                optimizer = torch.optim.SGD(self.packmodel.parameters(), lr=lr, weight_decay=self.lr_decay)
            else:
                optimizer = torch.optim.Adam(self.packmodel.parameters(), lr=lr,weight_decay=self.lr_decay)
        return optimizer

    def train(self, clip_model, t):
        self.model.to(self.device)
        self.model_old.to(self.device)
        self.packmodel.to(self.device)
        oldpackmodel = deepcopy(self.packmodel)
        self.temp_hook = clip_model.model.visual.transformer.resblocks[0].register_forward_hook(self.extract_feature())
        for image,_,_ in self.tr_dataloader:
            clip_model.model.encode_image(image.to(self.device))
        self.temp_hook.remove()
        if t!=self.old_task:
            self.model_old = deepcopy(self.model)
            self.model_old.train()
            freeze_model(self.model_old)  # Freeze the weights
            self.old_task = t
        self.optimizer = self._get_optimizer()
        self.pack.on_init_end(self.packmodel, t)
        # trian model
        if len(self.pack.masks) > t:
            self.pack.masks.pop()
        for e in range(self.nepochs):
            # Train
            if e < self.e_rep:
                for name, para in self.model.named_parameters():
                    if 'feature_net' in name:
                        para.requires_grad = False
                    else:
                        para.requires_grad = True
            else:
                for name, para in self.model.named_parameters():
                    if 'feature_net' in name:
                        para.requires_grad = True
                    else:
                        para.requires_grad = False
            if t == 0:
                self.train_epoch_rep(t, e, oldpackmodel, clip_model)
            else:
                if e < self.e_rep:
                    self.train_epoch_head(t, e, oldpackmodel, clip_model)
                else:
                    self.train_epoch_rep(t, e, oldpackmodel, clip_model)
            self.train_packnet(t, clip_model)
            self.pack.on_epoch_end(self.packmodel.feature_net,e,t)


            train_acc = self.eval(clip_model, t)
            if e % self.e_rep == self.e_rep -1:
                print('| Epoch {:3d} | acc={:5.1f}% | \n'.format(
                    e + 1, 100 * train_acc), end='')
        self.feature_data = []
        return self.fisher, train_acc
    def train_packnet(self, t, clip_model):
        self.packmodel.train()
        clip_model.model.eval()

        client_domain_feature = torch.tensor(self.preprocess(), dtype=torch.float)
        list_image_domain_features = list(DataLoader(TensorDataset(client_domain_feature),batch_size=50, shuffle=False))

        for index, batch in enumerate(self.tr_dataloader):
            images, _ ,labels = batch
            images, labels = images.to(self.device), labels.to(self.device)

            domain_feature = self.packmodel(list_image_domain_features[index][0].to(self.device))
            mean_domain_feature = domain_feature.mean(dim=0, keepdim=True)
            _mean_domain_features = mean_domain_feature.repeat_interleave(len(clip_model.labels), dim=0)
            text_features = clip_model._get_text_features(_mean_domain_features.half())
            image_features = clip_model.model.encode_image(images).half()
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            logit_scale = clip_model.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()

            loss = F.cross_entropy(logits_per_image, labels)

            self.pack_optimizer.zero_grad()
            loss.backward()
            self.pack.on_after_backward(self.packmodel.feature_net,t)
            self.pack_optimizer.step()

    def train_epoch_head(self, t, epoch,oldpackmodel, clip_model):
        self.model.train()
        clip_model.model.eval()

        client_domain_feature = torch.tensor(self.preprocess(), dtype=torch.float)
        list_image_domain_features = list(DataLoader(TensorDataset(client_domain_feature),batch_size=50, shuffle=False))

        for index, batch in enumerate(self.tr_dataloader):
            images, _, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            # Forward current model

            pre_image_features = clip_model.model.encode_image(images).half()
            # old model label
            pre_domain_feature = self.model_old(list_image_domain_features[index][0].to(self.device))
            pre_mean_domain_feature = pre_domain_feature.mean(dim=0, keepdim=True)
            _pre_mean_domain_features = pre_mean_domain_feature.repeat_interleave(len(clip_model.labels), dim=0)
            pre_text_features = clip_model._get_text_features(_pre_mean_domain_features.half())

            pre_image_features = pre_image_features / pre_image_features.norm(dim=1, keepdim=True)
            pre_text_features = pre_text_features / pre_text_features.norm(dim=1, keepdim=True)
            pre_logit_scale = clip_model.model.logit_scale.exp()
            pre_logits_per_image = pre_logit_scale * pre_image_features @ pre_text_features.t()
            # cur model label
            cur_domain_feature = self.model(list_image_domain_features[index][0].to(self.device))
            cur_mean_domain_feature = cur_domain_feature.mean(dim=0, keepdim=True)
            _cur_mean_domain_features = cur_mean_domain_feature.repeat_interleave(len(clip_model.labels), dim=0)
            cur_text_features = clip_model._get_text_features(_cur_mean_domain_features.half())

            cur_text_features = cur_text_features / cur_text_features.norm(dim=1, keepdim=True)
            cur_logits_per_image = pre_logit_scale * pre_image_features @ cur_text_features.t()
            
            # self.model.zero_grad()
            self.optimizer.zero_grad()
            self.model.zero_grad()
            clip_model.model.zero_grad()
            memoryloss = MultiClassCrossEntropy(pre_logits_per_image, cur_logits_per_image)
            memoryloss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            self.model.zero_grad()
            clip_model.model.zero_grad()
            # store_grad(self.model.parameters,grads, self.grad_dims,0)
            # image_features = clip_model.model.encode_image(images).half()
            cur_domain_feature = self.model(list_image_domain_features[index][0].to(self.device))
            cur_mean_domain_feature = cur_domain_feature.mean(dim=0, keepdim=True)
            _cur_mean_domain_features = cur_mean_domain_feature.repeat_interleave(len(clip_model.labels), dim=0)
            cur_text_features = clip_model._get_text_features(_cur_mean_domain_features.half())

            cur_text_features = cur_text_features / cur_text_features.norm(dim=1, keepdim=True)
            cur_logits_per_image = pre_logit_scale * pre_image_features @ cur_text_features.t()

            loss = F.cross_entropy(cur_logits_per_image, labels)

            ## 根据这个损失计算梯度，变换此梯度

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
    def train_epoch_rep(self, t, epoch,oldpackmodel, clip_model):
        self.model.train()
        clip_model.model.train()
        self.packmodel.train()
        clip_model.model.eval()
        self.model_old.eval()

        client_domain_feature = torch.tensor(self.preprocess(), dtype=torch.float)
        list_image_domain_features = list(DataLoader(TensorDataset(client_domain_feature),batch_size=50, shuffle=False))

        # Loop batches
        for index, batch in enumerate(self.tr_dataloader):
            # Forward current model
            images, _, labels = batch
            images, labels = images.to(self.device), labels.to(self.device) 
 
            self.optimizer.zero_grad()
            self.model.zero_grad()
            clip_model.model.zero_grad()
            grads = torch.Tensor(sum(self.grad_dims), 2+t)
            grads = grads.to(self.device)
            if t > 0:
                pre_image_features = clip_model.model.encode_image(images).half()
                
                pre_domain_feature = self.model_old(list_image_domain_features[index][0].to(self.device))
                pre_mean_domain_feature = pre_domain_feature.mean(dim=0, keepdim=True)
                _pre_mean_domain_features = pre_mean_domain_feature.repeat_interleave(len(clip_model.labels), dim=0)
                pre_text_features = clip_model._get_text_features(_pre_mean_domain_features.half())

                cur_domain_feature = self.model(list_image_domain_features[index][0].to(self.device))
                cur_mean_domain_feature = cur_domain_feature.mean(dim=0, keepdim=True)
                _cur_mean_domain_features = cur_mean_domain_feature.repeat_interleave(len(clip_model.labels), dim=0)
                cur_text_features = clip_model._get_text_features(_cur_mean_domain_features.half())


                pre_image_features = pre_image_features / pre_image_features.norm(dim=1, keepdim=True)
                pre_text_features = pre_text_features / pre_text_features.norm(dim=1, keepdim=True)
                cur_text_features = cur_text_features / cur_text_features.norm(dim=1, keepdim=True)
                
                pre_logit_scale = clip_model.model.logit_scale.exp()
                pre_logits_per_image = pre_logit_scale * pre_image_features @ pre_text_features.t()
                cur_logits_per_image = pre_logit_scale * pre_image_features @ cur_text_features.t()
                pre_loss =MultiClassCrossEntropy(pre_logits_per_image, cur_logits_per_image)

                pre_loss.backward()
                store_grad(self.model.feature_net.parameters, grads, self.grad_dims, 0)
                if t >= self.select_grad_num:
                    t = self.select_grad_num -1
                for i in range(t):
                    self.model.zero_grad()
                    self.optimizer.zero_grad()
                    clip_model.model.zero_grad()

                    temppackmodel = deepcopy(oldpackmodel).to(self.device)
                    temppackmodel.train()
                    self.pack.apply_eval_mask(task_idx=i, model=temppackmodel.feature_net)
                    # pre_image_features = clip_model.model.encode_image(images).half()
                    pre_domain_feature = self.model(list_image_domain_features[index][0].to(self.device))
                    pre_mean_domain_feature = pre_domain_feature.mean(dim=0, keepdim=True)
                    _pre_mean_domain_features = pre_mean_domain_feature.repeat_interleave(len(clip_model.labels), dim=0)
                    pre_text_features = clip_model._get_text_features(_pre_mean_domain_features.half())


                    with torch.no_grad():
                        pre_image_features = clip_model.model.encode_image(images).half()

                        temp_domain_feature = temppackmodel(list_image_domain_features[index][0].to(self.device))
                        temp_mean_domain_feature = temp_domain_feature.mean(dim=0, keepdim=True)
                        _temp_mean_domain_features = temp_mean_domain_feature.repeat_interleave(len(clip_model.labels), dim=0)
                        temp_text_features = clip_model._get_text_features(_temp_mean_domain_features.half())

                        pre_image_features = pre_image_features / pre_image_features.norm(dim=1, keepdim=True)
                        temp_text_features = temp_text_features / temp_text_features.norm(dim=1, keepdim=True)
                        pre_logit_scale = clip_model.model.logit_scale.exp()
                        temp_logits_per_image = pre_logit_scale * pre_image_features @ temp_text_features.t()
                    pre_text_features = pre_text_features / pre_text_features.norm(dim=1, keepdim=True)
                    pre_logits_per_image = pre_logit_scale * pre_image_features @ pre_text_features.t()
                    
                    memoryloss = MultiClassCrossEntropy(pre_logits_per_image, temp_logits_per_image)
                    memoryloss.backward()
                    store_grad(self.model.feature_net.parameters, grads, self.grad_dims, i+1)
                    del temppackmodel
                ## 求出每个分类器算出来的梯度

            image_features = clip_model.model.encode_image(images).half()

            domain_feature = self.model(list_image_domain_features[index][0].to(self.device))
            mean_domain_feature = domain_feature.mean(dim=0, keepdim=True)
            _mean_domain_features = mean_domain_feature.repeat_interleave(len(clip_model.labels), dim=0)
            text_features = clip_model._get_text_features(_mean_domain_features.half())

            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            logit_scale = clip_model.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()

            loss = F.cross_entropy(logits_per_image, labels)
            ## 根据这个损失计算梯度，变换此梯度

            # Backward
            self.model.zero_grad()
            clip_model.model.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            if t > 0:
                # copy gradient
                store_grad(self.model.feature_net.parameters, grads, self.grad_dims, t+1)
                taskl = [i for i in range(t+2)]
                indx = torch.LongTensor(taskl[:-1]).to(self.device)
                errindx = torch.LongTensor(0).to(self.device)
                dotp = torch.mm(grads[:, 1].unsqueeze(0),
                                grads.index_select(1, indx))
                if (dotp < 0).sum() != 0:
                    project2cone2(grads[:, t+1].unsqueeze(1),
                                  grads.index_select(1, indx), grads.index_select(1,errindx))
                    # copy gradients back
                    overwrite_grad(self.model.feature_net.parameters, grads[:, t+1],
                                   self.grad_dims)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=50, norm_type=2)
            self.optimizer.step()
        return

    def moretrain(self, t, clip_model):
        self.packmodel.to(self.device)
        for e in range(self.nepochs):
            self.train_packnet(t, clip_model)
            self.pack.on_epoch_end(self.packmodel.feature_net, e, t)

    def eval(self, clip_model, t, train=True):
        self.model.eval()
        clip_model.model.eval()
        if train:
            dataloaders = self.tr_dataloader
        client_domain_feature = torch.tensor(self.preprocess(), dtype=torch.float)
        list_image_domain_features = list(DataLoader(TensorDataset(client_domain_feature),batch_size=50, shuffle=False))
        total = 0
        correct = 0
        # Loop batches
        with torch.no_grad():
            for index, batch in enumerate(dataloaders):
                image, _, label = batch
                image, label = image.to(self.device), label.to(self.device)
                # Forward
                image_features = clip_model.model.encode_image(image).half()
                
                domain_feature = self.model(list_image_domain_features[index][0].to(self.device))
                mean_domain_feature = domain_feature.mean(dim=0, keepdim=True)
                _mean_domain_features = mean_domain_feature.repeat_interleave(len(clip_model.labels), dim=0)
                text_features = clip_model._get_text_features(_mean_domain_features.half())

                similarity = clu.get_similarity(image_features, text_features)

                _, indices = similarity.topk(1)
                total += len(label)
                pred = torch.squeeze(indices)
                res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
                res = res.cpu().numpy()
                correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])

        return correct / total

    def criterion(self, t):
        # Regularization for all previous tasks
        loss_reg = 0
        if t > 0:
            for (name, param), (_, param_old) in zip(self.model.feature_net.named_parameters(), self.model_old.feature_net.named_parameters()):
                loss_reg += torch.sum(self.fisher[name] * (param_old - param).pow(2)) / 2
        return self.lamb * loss_reg


def LongLifeTrain(args, clip_model, appr: Appr, aggNum, from_kb, idx):
    print('cur round :' + str(aggNum)+'  cur client:' + str(idx))
    # acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.half32)
    # lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.half32)
    t = aggNum // args.round
    print('cur task:' + str(t))
    r = aggNum % args.round
    # for t, ncla in taskcla:

    print('*' * 100)
    # print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    # Get data
    task = t

    # Train
    fisher, acc = appr.train(clip_model, task)
    print('-' * 100)
    return appr.model.state_dict(),fisher,acc,0

def LongLifeTest(args, appr, t, testdatas, aggNum, writer):
    t = aggNum // args.round
    r = aggNum % args.round
    # for u in range(t + 1):
    #     xtest = testdatas[u][0].to(self.device)
    #     ytest = (testdatas[u][1] - u * 10).to(self.device)
    #     test_loss, test_acc = appr.eval(u, xtest, ytest)
    #     print('>>> Test on task {:2d} : loss={:.3f}, acc={:5.1f}% <<<'.format(u, test_loss,
    #                                                                           100 * test_acc))
    #     acc[0, u] = test_acc
    #     lss[0, u] = test_loss
    # # Save
    # mean_acc = np.mean(acc[0, :t+1])
    # mean_lss = np.mean(lss[0, :t])
    # print('Average accuracy={:5.1f}%'.format(100 * np.mean(acc[0, :t+1])))
    # print('Average loss={:5.1f}'.format(np.mean(lss[0, :t+1])))
    # print('Save at ' + args.output)
    # if r == args.round - 1:
    #     writer.add_scalar('task_finish_and _agg', mean_acc, t + 1)
    # # np.savetxt(args.agg_output + 'aggNum_already_'+str(aggNum)+args.log_name, acc, '%.4f')
    # return mean_lss, mean_acc

# def main():
#     # cifar100 = Cifar100Task('../data',batch_size=900,num_clients=5,cur_client=4,task_num=10,isFed=True)
#     cifar100 = Cifar100Task('../data/cifar-100-python', batch_size=4500, task_num=10, num_clients=5, cur_client=0,
#                       isFed=True)
#     TaskDatas = cifar100.getDatas()
#     net = network.RepTail([3, 32, 32]).to(self.device)
#
#
# if __name__ == "__main__":
#     main()