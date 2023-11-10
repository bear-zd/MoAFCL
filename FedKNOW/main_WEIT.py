import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
#python main_FedKNOW.py --alg=WEIT --dataset=officehome --net=ViT-B/16 --root_dir=/mnt/sda/zd/data/splitdata  --num_users=10 --frac=1 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=4 --epoch=150  --local_ep=8  --gpu=0 --batch 100
from utils.options import args_parser

from models.test import  test_img_local_all_WEIT
from single.ContinualLearningMethod.WEIT import Appr,LongLifeTrain
from models.Nets import  WeITAdapter
from utils.dataload import DomainDataset, get_data
from models.clip import FedWeITClip

import time
def img_param_init(args):
    dataset = args.dataset
    if dataset == 'pacs':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
        args.num_classes = 7
    elif dataset == 'officehome':
        domains = ['Art', 'Clipart', 'Product', 'Real World']
        args.numclasses = 65
    else :
        raise BaseException(f"{dataset} dataset not defined!")
    args.wd = 1e-4
    args.lambda_l1 = 1e-3
    args.lambda_l2 = 1
    args.lambda_mask = 0
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.domains = domains
    return args

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args = img_param_init(args)
    
    lens = np.ones(args.num_users)
    server_model = FedWeITClip(args.net,args.device)
    dataloader: DomainDataset = get_data(args.dataset)(args, server_model.preprocess)
    train_loaders, test_loaders, labels = dataloader.get_dataloader(0)
    server_model.labels = labels
    server_model.init_prompt()

    # lens = np.ones(args.num_users)
    # if 'cifar' in args.dataset or args.dataset == 'mnist' or 'miniimagenet' in args.dataset or 'FC100' in args.dataset or 'Corn50' in args.dataset:
    #     dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    #     for idx in dict_users_train.keys():
    #         np.random.shuffle(dict_users_train[idx])

    print(args.alg)
    write = SummaryWriter('./log/WEIT_' + args.dataset+'_'+'round' + str(args.round) + '_frac' + str(args.frac) + '_model_'+args.net)
    # build model
    # net_glob = get_model(args)
    net_glob = WeITAdapter(args.net, device=args.device)
    net_glob.train()
    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    w_glob_keys = [net_glob.weight_keys[i] for i in [j for j in range(len(net_glob.weight_keys))]]
    num_param_glob = 0
    num_param_local = 0
    for key in net_glob.state_dict().keys():
        num_param_local += net_glob.state_dict()[key].numel()
        print(num_param_local)
        if key in w_glob_keys:
            num_param_glob += net_glob.state_dict()[key].numel()
    percentage_param = 100 * float(num_param_glob) / num_param_local
    print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
        num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    print("learning rate, batch size: {}, {}".format(args.lr, args.local_bs))


    # generate list of local models for each user
    net_local_list = []
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict

    # training
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()
    task=-1
    apprs = [Appr(copy.deepcopy(net_glob).to(args.device), None,lr=args.lr, nepochs=args.local_ep, args=args) for i in range(args.num_users)]
    print(args.round)
    from_kb =[]
    for name,para in net_glob.named_parameters():
        if 'aw' in name:
            shape = np.concatenate([para.shape, [int(round(args.num_users * args.frac))]], axis=0)
            from_kb_l = np.zeros(shape)
            from_kb_l = torch.from_numpy(from_kb_l)
            from_kb.append(from_kb_l)
    w_glob=[]
    for iter in range(args.epochs):
        if iter % (args.round) == 0:
            task+=1
        w_agg=w_glob
        w_glob = []
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        w_keys_epoch = w_glob_keys
        times_in = []
        total_len = 0
        train_dataloaders= None
        if iter % args.round == 0:
            for i in range(args.num_users):
                apprs[i].model.set_knowledge(task,from_kb)
        for ind, idx in enumerate(idxs_users):
            glob_fisher = None
            start_in = time.time()
            
            #  tr_dataloaders = DataLoader(DatasetSplit(dataset_train[task],dict_users_train[idx][:args.m_ft]),batch_size=args.local_bs, shuffle=True)
            train_dataloaders, test_dataloaders, _ = dataloader.get_dataloader(task)
            w_local = []
            appr:Appr = apprs[idx]
            appr.set_sw(w_agg)
            appr.set_trData(train_dataloaders[idx])
            last = iter == args.epochs

            w_local, aws,loss, indd = LongLifeTrain(args,server_model ,appr,iter,from_kb,idx)
            if iter % args.round == args.round -1:
                from_kb = []
                for aw in aws:
                    shape = np.concatenate([aw.shape, [int(round(args.num_users * args.frac))]], axis=0)
                    from_kb_l = np.zeros(shape)
                    if len(shape) == 5:
                        from_kb_l[:, :, :, :, ind] = aw.cpu().detach().numpy()
                    else:
                        from_kb_l[:, :, ind] = aw.cpu().detach().numpy()
                    from_kb_l = torch.from_numpy(from_kb_l)
                    from_kb.append(from_kb_l)
            loss_locals.append(copy.deepcopy(loss.item()))
            total_len += lens[idx]
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for i in range(len(w_glob)):
                    w_glob[i] = w_glob[i] * lens[idx]
            else:
                for i in range(len(w_glob)):
                    w_glob[i] += w_local[i]*lens[idx]
            times_in.append(time.time() - start_in)
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # get weighted average for global weights
        for i in range(len(w_glob)):
            w_glob[i] = torch.div(w_glob[i], total_len)
        # print(w_glob)
        if iter % args.round == args.round-1:
            for i in range(args.num_users):
                if len(apprs[i].pre_weight['aw']) < task+1:
                    print("client " + str(i) + " not train")
                    train_dataloaders, test_dataloaders, _ = dataloader.get_dataloader(task)
                    apprs[i].set_sw(w_glob)
                    apprs[i].set_trData(train_dataloaders[i])
                    # LongLifeTrain(args, server_model,apprs[i], iter, from_kb, i) # I don't know why there still train
            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
            print('task ' + str(task) + ' finish train')
            with torch.no_grad():
                acc_test, total_num = test_img_local_all_WEIT(server_model,apprs, args, test_dataloaders,device=args.device)
            # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
            print('Round {:3d}, task {},Test accuracy: {}'.format(iter, task, acc_test/total_num))

    # print('Average accuracy final 10 rounds: {}'.format(accs10))
    # if args.alg == 'fedavg' or args.alg == 'prox':
    #     print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
    end = time.time()
    print(end - start)
    print(times)
    print(accs)
    base_dir = './save/WEIT/accs_WEIT_lambda_'+str(args.lamb) +str('_') + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) + '_iterFinal' + '_frac_'+str(args.frac)+ '_model_'+args.model+'.csv'
    user_save_path = base_dir
    accs = np.array(accs)
    accs = pd.DataFrame(accs, columns=['accs'])
    accs.to_csv(base_dir, index=False)
