nohup python FedKNOW/main_FedKNOW.py --alg fedKNOW  --dataset=officehome --net=ViT-B/16 --root_dir=/mnt/sda/zd/data/OF10-10-1000-2023 --num_users=10 --frac=1 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=10 --round=1  --local_ep=10  --gpu=2 --batch 50 > FedKNOW10.out 2>&1 &

nohup python FedKNOW/main_FedKNOW.py --alg fedKNOW  --dataset=officehome --net=ViT-B/16 --root_dir=/mnt/sda/zd/data/OF10-10-1000-2023 --num_users=10 --frac=1 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=4 --epoch=4 --round=1  --local_ep=10  --gpu=2 --batch 50 > FedKNOW4.out 2>&1 &

nohup python FedKNOW/main_WEIT.py --alg=WEIT --dataset=officehome --net=ViT-B/16 --root_dir=/mnt/sda/zd/data/OF10-10-1000-2023 --num_users=10 --frac=1 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=4 --epoch=4 --round=1  --local_ep=10  --gpu=1 --batch 50 > weit4.out 2>&1 &

nohup python FedKNOW/main_WEIT.py --alg=WEIT --dataset=officehome --net=ViT-B/16 --root_dir=/mnt/sda/zd/data/OF10-10-1000-2023 --num_users=10 --frac=1 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=10 --round=1  --local_ep=10  --gpu=1 --batch 50 > weit.out 2>&1 &