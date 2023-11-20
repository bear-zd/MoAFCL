nohup python FedKNOW/main_FedKNOW.py --alg fedKNOW  --dataset=officehome --net=ViT-B/16 --root_dir=/mnt/sda/zd/data/OF10-10-1000-2023 --num_users=10 --frac=1 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=10 --round=1  --local_ep=10  --gpu=2 --batch 50 > FedKNOW10.out 2>&1 &

nohup python FedKNOW/main_FedKNOW.py --alg fedKNOW  --dataset=officehome --net=ViT-B/16 --root_dir=/mnt/sda/zd/data/OF10-10-1000-2023 --num_users=10 --frac=1 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=4 --epoch=4 --round=1  --local_ep=10  --gpu=2 --batch 50 > FedKNOW4.out 2>&1 &

nohup python FedKNOW/main_WEIT.py --alg=WEIT --dataset=officehome --net=ViT-B/16 --root_dir=/mnt/sda/zd/data/OF10-10-1000-2023 --num_users=10 --frac=1 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=4 --epoch=4 --round=1  --local_ep=10  --gpu=1 --batch 50 > weit4.out 2>&1 &

nohup python FedKNOW/main_WEIT.py --alg=WEIT --dataset=officehome --net=ViT-B/16 --root_dir=/mnt/sda/zd/data/OF10-10-1000-2023 --num_users=10 --frac=1 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=10 --round=1  --local_ep=10  --gpu=1 --batch 50 > weit.out 2>&1 &

# "tree","golf_club","squirrel","dog","whale","spreadsheet","snowman","tiger","table","shoe","windmill","submarine","truck","feather","bird","spider","strawberry","nail","beard","bread","train","watermelon","zebra","sheep","elephant","teapot","eye","mushroom","sea_turtle","sword","streetlight","lighthouse","bridge","owl","horse","penguin","pond","sock","snorkel","helicopter","snake","butterfly","umbrella","river","fish","van","grapes","hot_air_balloon","wine_glass","teddy-bear","speedboat","sun","swan","bicycle","brain","bracelet","tornado","flower","stairs","cup","steak","vase","tractor","wristwatch","stethoscope","suitcase","triangle","parrot","zigzag","ice_cream","mug","beach","cat","raccoon","garden","monkey","shark","animal_migration","lion","saxophone","asparagus","tent","firetruck","The_Eiffel_Tower","hand","spoon","squiggle","palm_tree","octopus","toaster","skateboard","dumbbell","headphones","mountain","bottlecap","hexagon","pig","toilet","washing_machine","frog"

nohup python FedKNOW/main_FedKNOW.py --alg fedKNOW  --dataset=domainnetsub --net=ViT-B/16 --root_dir=/root/autodl-tmp/DN20-10-2000-2023 --num_users=20 --frac=1 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=10 --round=1  --local_ep=5  --gpu=0 --batch 50 > FedKNOW10DNsub5local.out 2>&1 &

nohup python FedKNOW/main_WEIT.py --alg fedWeIT  --dataset=domainnetsub --net=ViT-B/16 --root_dir=/root/autodl-tmp/DN20-10-2000-2023 --num_users=20 --frac=1 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=10 --round=1  --local_ep=5  --gpu=0 --batch 50 > FedWeIT10DNsub5local.out 2>&1 &

nohup python FedKNOW/main_FedKNOW.py --alg fedKNOW  --dataset=domainnetsub --net=ViT-B/16 --root_dir=/root/autodl-tmp/DN20-10-2000-2023 --num_users=20 --frac=1 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=10 --round=1  --local_ep=10  --gpu=0 --batch 50 > FedKNOW10DNsub10local.out 2>&1 &


conda create -n FedKNOW python=3.8
conda activate FedKNOW
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
nohup python main.py --dataset officehome --root_dir /mnt/sda/zd/data/OF10-10-1000-2023  --batch 100  --n_experts=5  --device cuda:0 --inner_iter 5 --n_task 10 --seed 2023 > nota.out 2>&1 &
nohup python main.py --dataset domainnetsub --root_dir ~/autodl-tmp/  --batch 100  --n_experts=10  --device cuda:0 --inner_iter 5 --n_task 10 --n_clients 20 --seed 2023 > s.out 2>&1 &

nohup python -u fedclip.py --dataset domainnetsub --root_dir ~/autodl-tmp/DN20-10-2000-2023  --batch 75  --device cuda:0 --inner_iter 5 --n_task 20 --n_clients 10 --seed 2023 > fedclipDNfull.out 2>&1 &

 nohup python -u  fedclip.py --dataset officehome --root_dir /root/autodl-tmp/OF10-10-1000-2023  --batch 100 > fedclipOF-5.out 2>&1 &

# nohup python main.py --dataset officehome --root_dir /mnt/sda/zd/data/OF10-10-1000-2023/  --batch 100  --n_experts=5 --extract_layer 5  --device cuda:0 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 10 --rand 0 > lapOF1e-2.out 2>&1 &

# nohup python main.py --dataset domainnetsub --root_dir /mnt/sda/zd/data/DN20-10-2000-2023/  --batch 100  --n_experts=8 --extract_layer 5  --device cuda:0 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 20 --rand 0 > lapDN1e-2.out 2>&1 &

# nohup python main.py --dataset officehome --root_dir /mnt/sda/zd/data/OF10-10-1000-2023/  --batch 100  --n_experts=5 --extract_layer 5  --device cuda:1 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 10 --rand 0 > lapOF1e-1.out 2>&1 &

nohup python main.py --dataset domainnetsub --root_dir ~/autodl-tmp/DN20-10-2000-2023/  --batch 100  --n_experts=8 --extract_layer 5  --device cuda:1 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 20 --rand 0 > lapDN1e-1.out 2>&1 &

# nohup python main.py --dataset officehome --root_dir /mnt/sda/zd/data/OF10-10-1000-2023/  --batch 100  --n_experts=5 --extract_layer 5  --device cuda:2 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 10 --rand 0 > lapOF1e0.out 2>&1 &

nohup python main.py --dataset domainnetsub --root_dir /mnt/sda/zd/data/DN20-10-2000-2023/  --batch 100  --n_experts=8 --extract_layer 5  --device cuda:1 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 20 --rand 0 > lapDN5.out 2>&1 &
# server site
nohup python main.py --dataset officehome --root_dir /mnt/sda/zd/data/OF10-10-1000-2023/  --batch 100  --n_experts=5 --extract_layer 5  --device cuda:0 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 10 --rand 0 > lapOF5.out 2>&1 &

# nohup python main.py --dataset domainnetsub --root_dir ~/autodl-tmp/DN20-10-2000-2023/  --batch 100  --n_experts=8 --extract_layer 5  --device cuda:0 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 20 --rand 0 > lapDNNo.out 2>&1 &

nohup python main.py --dataset adaptiope --root_dir  /mnt/sda/zd/data/AD10-10-1000-2023 --batch 100  --n_experts=3 --extract_layer 1  --device cuda:0 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 10 --rand 0 > ADel1.out 2>&1 &
nohup python main.py --dataset adaptiope --root_dir  /mnt/sda/zd/data/AD10-10-1000-2023 --batch 100  --n_experts=3 --extract_layer 3  --device cuda:1 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 10 --rand 0 > ADel3.out 2>&1 &
nohup python main.py --dataset adaptiope --root_dir  /mnt/sda/zd/data/AD10-10-1000-2023 --batch 100  --n_experts=3 --extract_layer 5  --device cuda:2 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 10 --rand 0 > ADel5.out 2>&1 &
nohup python main.py --dataset adaptiope --root_dir  /mnt/sda/zd/data/AD10-10-500-2023 --batch 100  --n_experts=3 --extract_layer 7  --device cuda:0 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 10 --rand 0 > ADel7.out 2>&1 &
nohup python main.py --dataset adaptiope --root_dir  /mnt/sda/zd/data/AD10-10-500-2023 --batch 100  --n_experts=3 --extract_layer 9  --device cuda:1 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 10 --rand 0 > ADel9.out 2>&1 &
nohup python main.py --dataset adaptiope --root_dir  /mnt/sda/zd/data/AD10-10-500-2023 --batch 100  --n_experts=3 --extract_layer 11  --device cuda:2 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 10 --rand 0 > ADel11.out 2>&1 &

# nohup python main.py --dataset adaptiope --root_dir  /mnt/sda/zd/data/AD10-10-500-2023 --batch 100  --n_experts=5 --extract_layer 7  --device cuda:0 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 10 --rand 0 > ADel7.out 2>&1 &
# nohup python main.py --dataset adaptiope --root_dir  /mnt/sda/zd/data/AD10-10-500-2023 --batch 100  --n_experts=5 --extract_layer 9  --device cuda:1 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 10 --rand 0 > ADel9.out 2>&1 &
# nohup python main.py --dataset adaptiope --root_dir  /mnt/sda/zd/data/AD10-10-500-2023 --batch 100  --n_experts=5 --extract_layer 11  --device cuda:2 --inner_iter 5 --n_task 10 --seed 2023 --n_clients 10 --rand 0 > ADel11.out 2>&1 &