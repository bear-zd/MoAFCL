# FMC
Federated Learning CLIP with mixture of experts(adapter)

https://github.com/davidmrau/mixture-of-experts MoE，引入到model/moe.py

https://github.com/microsoft/PersonalizedFL/tree/main/fedclip FedCLIP

关于代码详细的问题可以直接问

## install

```bash
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

## 进展

MoE + CLIP + Continual Learning 代码完成并尝试训练。


划分数据集需要参考utils/splitdata.py

运行我们的方法：（具体参数在argument.py，数据的分割和数据结构参照utils/splitdata.py）
> python main.py --dataset officehome --root_dir /mnt/sda/zd/data/splitdata/  --batch 100  --logdir t2.log

运行fedclip:
> python fed_at_clip.py --dataset officehome --root_dir /mnt/sda/zd/data/splitdata/  --batch 100  --logdir t2.log

运行FedWeIT:
> python main_WEIT.py --alg=WEIT --dataset=officehome --net=ViT-B/16 --root_dir=/mnt/sda/zd/data/splitdata  --num_users=10  --frac=0.1 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=4   --local_ep=1  --gpu=1 --batch 100

大致流程：
- 初始启动时，基于每个客户端的特征进行聚类得到五个类别experts
- 对每个客户端进行训练之后根据所属的experts进行fedavg
- 完成初始启动之后：
   - 客户端到来时根据自己的数据特征进行experts选择
   - 每个客户端训练对应的experts并聚合
- 评估： 

### 工作描述

=== 我们的工作量

- Federated learning Continous learning with Mixture of Adapter(Expert) 
- 数据集划分

=== baseline工作量

- FedCLIP 迁移：已经完成了

- 联邦持续学习迁移到CLIP和我们的数据上



### 未完成的工作

- 测试结果



