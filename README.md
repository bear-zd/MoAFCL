# FMC
Federated Learning CLIP with mixture of experts(adapter)

https://github.com/davidmrau/mixture-of-experts MoE，引入到model/moe.py

https://github.com/microsoft/PersonalizedFL/tree/main/fedclip FedCLIP

关于代码详细的问题可以直接问

## 进展

MoE + CLIP + Continual Learning 代码完成并尝试训练。

```python
server_model: ClipModelMA = ClipModelMA(args.net, n_experts=args.n_experts, device=args.device)  # load the server data
train_loaders, test_loaders, labels = get_data(args.dataset)(args, server_model.preprocess).get_dataloader()

server_model.init_MoE(**PARA_DICT[args.dataset]) # expert number already saved in the server_model.

test_acc_list, train_acc_list = [], []
for i in range(args.n_clients):
    server_model.labels = labels[i]
    image_adapter = Adapter(**PARA_DICT[args.dataset], label=f"client{i}")

    optimizer = optim.Adam(params=image_adapter.parameters(), lr=args.lr, betas=(
        args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay) 
    server_model.model.to(args.device)
    image_adapter.to(args.device)
    for epoch in tqdm(range(args.inner_iter)):
        train_client(server_model, image_adapter, train_loaders[i], optimizer ,args.device)
        train_acc = test_client(server_model, image_adapter, train_loaders[i],  args.device)
        test_acc = test_client(server_model, image_adapter, test_loaders[i],  args.device)

    communicate(server_model, image_adapter, train_loaders[i], args.device)

    test_acc_list.append(test_server(server_model, test_loaders[i], args.device))
    train_acc_list.append(test_server(server_model, train_loaders[i], args.device))

```
划分数据集需要参考utils/splitdata.py

运行我们的方法：（具体参数在argument.py，数据的分割和数据结构参照utils/splitdata.py）
> python main.py --dataset officehome --root_dir /mnt/sda/zd/data/splitdata/  --batch 100  --logdir t2.log

运行fedclip:
> python fed_at_clip.py --dataset officehome --root_dir /mnt/sda/zd/data/splitdata/  --batch 100  --logdir t2.log

运行FedWeIT:
> python main_WEIT.py --alg=WEIT --dataset=officehome --net=ViT-B/16 --root_dir=/mnt/sda/zd/data/splitdata  --num_users=10 > --frac=0.1 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=4   --local_ep=1  --gpu=1 --batch 100

大致流程：每个客户端训练自己的模型（**todo:拉取这部分还没有搞**），之后把自己的Adapter上传到服务器的专家模型中，其中置换掉最老的模型（这里可以上一些内存调度算法像什么LRU这些)

之后模型获得置换模型的索引并将其作为one-hot向量，利用客户端的数据进行训练从而不需要发送标签，保护用户隐私（**todo:可以添加一些差分噪声或者使用一些较新的方法：[FedCR](https://proceedings.mlr.press/v202/zhang23w/zhang23w.pdf)**）



### 工作描述

=== 我们的工作量

- Federated learning Continous learning with Mixture of Adapter(Expert) 
- 数据集划分

=== baseline工作量

- FedCLIP 迁移：已经完成了

- 联邦持续学习迁移到CLIP和我们的数据上



### 未完成的工作

- 论文我们方法的优化工作（比较复杂，需要微调和多尝试）

  - 完成整体pipeline

 - 联邦持续学习的相关baseline迁移

   - （高优先级）FedKNOW 迁移实现， FedWeIT检验


