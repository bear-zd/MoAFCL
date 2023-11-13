import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.options import args_parser
# from utils.train_utils import get_data, get_model
from models.Nets import KNOWAdapter
from models.test import test_img_local_all_KNOW
from single.ContinualLearningMethod.FedKNOW import Appr,LongLifeTrain
import time
from models.Packnet import PackNet
from models.clip import FedKNOWClip
from utils.dataload import DomainDataset, get_data
import random
def set_random_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def img_param_init(args):
    set_random_seed(2023)
    dataset = args.dataset
    if dataset == 'pacs':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
        args.num_classes = 7
    elif dataset == 'officehome':
        domains = ['Art', 'Clipart', 'Product', 'Real World']
        args.numclasses = 65
    elif dataset == 'domainnet':
        domains = ['clipart','infograph','painting','quickdraw','real','sketch']
        args.numclasses = 345
    elif dataset == 'domainnetsub':
        domains = ['clipart','infograph','painting','quickdraw','real','sketch']
        args.numclasses = 100
    else:
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
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    lens = np.ones(args.num_users)

    server_model = FedKNOWClip(args.net,args.device)
    dataloader: DomainDataset = get_data(args.dataset)(args, server_model.preprocess)
    train_loaders, test_loaders, labels = dataloader.get_dataloader(0)
    server_model.labels = labels
    server_model.init_prompt()

    print(args.alg)
    write = SummaryWriter('./log/FedKNOW_' + args.dataset+'_'+'round' + str(args.round) + '_frac' + str(args.frac) + '_model_'+args.net)
    # build model
    # net_glob = get_model(args)
    net_glob = KNOWAdapter(args.net)
    net_glob.train()
    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()] # adapter内的参数
    w_glob_keys = []
    # 改到这里存疑
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
    task = -1
    apprs = [Appr(copy.deepcopy(net_glob),
                  PackNet(args.task, local_ep=args.local_ep, local_rep_ep=args.local_local_ep, device=args.device,
                          prune_instructions=1 - args.store_rate), copy.deepcopy(net_glob), None, lr=args.lr,
                  nepochs=args.local_ep, args=args) for i in range(args.num_users)]
    print(args.round)


    for iter in range(args.epochs):
        if iter % (args.round) == 0:
            task += 1
        w_glob = {}
        fisher_glob = {}
        acc_locals = []
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        w_keys_epoch = w_glob_keys
        times_in = []
        total_len = 0
        train_dataloaders = None
        acc_train = []

        for ind, idx in enumerate(idxs_users):
            glob_fisher = None
            start_in = time.time()
            train_dataloaders, test_dataloaders, _ = dataloader.get_dataloader(task)
            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            for k in w_locals[idx].keys():
                if k not in w_glob_keys:
                    w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
            appr: Appr = apprs[idx]
            appr.set_model(net_local.to(args.device))
            appr.set_trData(train_dataloaders[idx])
            last = iter == args.epochs
            w_local, fisher, acc, indd = LongLifeTrain(args, server_model, appr, iter, None, idx)
            acc_locals.append(copy.deepcopy(acc))
            total_len += lens[idx]

            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_glob[key] * lens[idx]
                    w_locals[idx][key] = w_local[key]
            else:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    if key in w_glob_keys:
                        w_glob[key] += w_local[key] * lens[idx]
                    else:
                        w_glob[key] += w_local[key] * lens[idx]
                    w_locals[idx][key] = w_local[key]
            if args.lamb != 0:
                if len(fisher_glob) == 0:
                    fisher_glob = copy.deepcopy(fisher)
                    for k, key in enumerate(net_glob.feature_net.state_dict().keys()):
                        if 'running_var' not in key and 'running_mean' not in key and 'num_batches_tracked' not in key:
                            fisher_glob[key] = fisher[key] * lens[idx]
                            fisher_glob[key] = fisher[key] * lens[idx]
                else:
                    for k, key in enumerate(net_glob.feature_net.state_dict().keys()):
                        if 'running_var' not in key and 'running_mean' not in key and 'num_batches_tracked' not in key:
                            fisher_glob[key] += fisher[key] * lens[idx]
            times_in.append(time.time() - start_in)
        # loss_avg = sum(loss_locals) / len(loss_locals)
        # loss_train.append(loss_avg)
        acc_avg = sum(acc_locals) / len(acc_locals)
        acc_train.append(acc_avg)

        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)

        if args.lamb != 0:
            for k in net_glob.feature_net.state_dict().keys():
                if 'running_var' not in key and 'running_mean' not in key and 'num_batches_tracked' not in key:
                    fisher_glob[k] = torch.div(fisher_glob[k], total_len)
        if args.lamb != 0:
            for i in range(args.num_users):
                apprs[i].set_fisher(fisher_glob)
        w_local = net_glob.state_dict()
        for k in w_glob.keys():
            w_local[k] = w_glob[k]
        if args.epochs != iter:
            net_glob.load_state_dict(w_glob)

        if iter % args.round == args.round - 1:
            for i, appr in enumerate(apprs):
                if len(appr.pack.masks) <= task:
                    print('client ' + str(i) + ' more train')
                    train_dataloaders, test_dataloaders, _ = dataloader.get_dataloader(task)
                    appr.set_trData(train_dataloaders[i])
                    # appr.moretrain(task)
            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
            print('task ' + str(task) + ' finish train')
            with torch.no_grad():
                acc_test, total_num = test_img_local_all_KNOW(server_model, apprs, args, test_dataloaders, device=args.device)
            accs.append(acc_test)
            # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
            print('Round {:3d}, task {},Test accuracy: {}'.format(iter, task, acc_test/total_num))

    end = time.time()
    print(end - start)
    print(times)
    print(accs)
    base_dir = './save/FedKNOW/accs_FedKNOW_lambda_' + str(args.lamb) + str(
        '_') + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
        args.shard_per_user) + '_iterFinal' + '_frac_' + str(args.frac) + '_model_' + args.net + '.csv'
    user_save_path = base_dir
    accs = np.array(accs)
    accs = pd.DataFrame(accs, columns=['accs'])
    accs.to_csv(base_dir, index=False)
