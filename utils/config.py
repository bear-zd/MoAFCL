import random
import numpy as np
import torch
import logging


def img_param_init(args):
    dataset = args.dataset
    if dataset == 'pacs':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
        args.num_classes = 7
    elif dataset == 'officehome':
        domains = ['Art', 'Clipart', 'Product', 'Real World']
        args.numclasses = 65
    elif dataset == 'domainnetsub':
        domains = ['clipart','infograph','painting','quickdraw','real','sketch']
        args.numclasses = 100
    elif dataset == 'adaptiope':
        domains = ['synthetic', 'real_life', 'product_images']
        args.numclasses = 123
    else :
        logging.ERROR(f"{dataset} dataset not defined!")
        raise BaseException(f"{dataset} dataset not defined!")
    args.domains = domains
    return args


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
