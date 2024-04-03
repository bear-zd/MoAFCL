import argparse
def argparser():
    parser = argparse.ArgumentParser()
    # meta setting parameter
    parser.add_argument('--dataset', type=str, default='officehome')
    parser.add_argument('--root_dir', type=str, default='/data/')
    parser.add_argument('--net', type=str, default='ViT-B/16',
                        help='[RN50 | RN101 | RN50x4 | RN50x16 | RN50x64 | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px]')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--logdir', type=str, default=None, help='log save path')
    parser.add_argument('--device', type=str, default="cuda:1", help='set device info')
    parser.add_argument('--n_clients', type=int, default=10, help="number of clients")
    parser.add_argument('--n_adapters', type=int, default=5, help="number of adapter model")
    parser.add_argument('--inner_iter', type=int, default=1, help="number clients inner train iter times")
    parser.add_argument('--n_task', type=int, default=4, help="number of tasks")
    parser.add_argument('--rand', action='store_true', help="rand choice of MoE")
    parser.add_argument('--extract_layer', type=int, default=1, help="feature extract position")
    # data setting parameter
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    

    
    # optimizer parameter
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    args = parser.parse_args()
    return args