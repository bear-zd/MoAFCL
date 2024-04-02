# MoAFCL
Mixture of Adapters Federated Continous Learning

https://github.com/davidmrau/mixture-of-adapters MoE，model/moe.py

https://github.com/microsoft/PersonalizedFL/tree/main/fedclip FedCLIP

## install

```bash
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
## splitdata
Before running the code, do split data first.

> python utils/splitdata.py

## run

run MoAFCL
> python -u main.py --dataset officehome --root_dir xxx --batch 100  --n_adapters=5 --extract_layer 5  --device cuda:0 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 10 --rand 0 

run FedCLIP
> python -u fedclip.py --dataset officehome --root_dir xxx --batch 100  --n_adapters=5 --extract_layer 5  --device cuda:0 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 10 --rand 0 

run FedWeIT
> python -u FedKNOW/main_WEIT.py --alg fedWeIT  --dataset=officehome --net=ViT-B/16 --root_dir= xxx --num_users=10 --frac=1 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=10 --round=1  --local_ep=10  --gpu=0 --batch 100

run FedKNOW
> python -u FedKNOW/main_FedKNOW.py --alg fedKNOW  --dataset=officehome --net=ViT-B/16 --root_dir= xxx --num_users=10 --frac=1 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=10 --round=1  --local_ep=10  --gpu=0 --batch 100

Detailed run shell code can be found in run.sh
