import utils.cliputils as clu
from utils.dataload import *
from model.clip import ClipModelMA, Client
import torch.optim as optim
from tqdm import tqdm
from torch.nn.functional import binary_cross_entropy_with_logits
import torch
from typing import List
import itertools
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Softmax
from utils.cliputils import freeze_param, unfreeze_param

def add_laplace_noise(vector, sensitivity=1, epsilon=100,device=None):
    b = sensitivity / epsilon
    noise = np.random.laplace(scale=b, size=vector.shape)    
    noisy_vector = vector + torch.tensor(noise).to(device)
    return noisy_vector.float()

def train_server(clip_model: ClipModelMA, clients: List[Client], task, device):
    clip_model.MoE = clip_model.MoE.to(device)

    optimizer = optim.SGD(clip_model.MoE.gating.parameters(), lr=1.5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [200,400], gamma=0.5)
    clip_model.MoE.train()
    freeze_param(clip_model.MoE.adapters)
    clip_model.MoE.adapters.eval()

    
    shape = 768 if "B" in clip_model.model_name else 1024
    temp_data = torch.tensor(np.stack([i.preprocess() for i in clients]).reshape(-1, shape), dtype=torch.float)
    # temp_label_data = torch.tensor(list(itertools.chain.from_iterable([[i.count_dict]*len(i.preprocess()) for i in clients])), dtype=torch.long)
    temp_label_data = torch.tensor(list(itertools.chain.from_iterable([[i.count_dict]*len(i.preprocess()) for i in clients])), dtype=torch.float)

    # temp_label_data = torch.tensor([[]])
    # print(temp_data.shape, temp_label_data.shape)
    dataset = TensorDataset(temp_data, temp_label_data)
    dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=True)
    correct = 0
    all = 0
    softmax = Softmax(-1)
    logging.info(f"Server start to train MoE !")
    for epoch in range(500):
        for batch in dataloader:
            data, label = batch
            # print(label)

            data, label = data.to(device), label.to(device)
            data = add_laplace_noise(data, device=device)
            _ , loss_gate, logits = clip_model.MoE(data, train_gate=True)
            
            # one_hot_label = torch.ones_like(logits, device=device)*0.2
            # one_hot_label[:, label] = 1
            # loss_label = binary_cross_entropy_with_logits(logits, label )
            loss_label = binary_cross_entropy_with_logits(logits, label) 

            loss =  2*loss_gate + loss_label
            # print(f"the loss of gate:{loss_gate.item()}, the loss of label {loss_label.item()}")

            all += len(data)
            pred = torch.argmax(logits, -1).detach()
            real = torch.argmax(label, -1).detach()
            res = torch.cat([pred.view(-1, 1), real.view(-1, 1)], dim=1)
            res = res.cpu().numpy()
            
            correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1)%100 == 0:
            print(f"epoch {epoch} the trainning set MoE acc : {correct/all}")
    torch.save(clip_model.MoE.gating.state_dict(), f"save/gatingAD{task}.pkl" )
    unfreeze_param(clip_model.MoE.adapters)



@torch.no_grad()
def test_server(clip_model: ClipModelMA, data_loader: DataLoader, server_data, device):
    clip_model.model.eval()
    clip_model.MoE.eval()
    total = 0
    correct = 0
    
    data_feature = torch.tensor(server_data.preprocess(),dtype=torch.float)
    # print(data_feature[0].shape, type(data_feature[0]))
    dataset = TensorDataset(data_feature)
    list_feature_dataloader = list(DataLoader(dataset=dataset, batch_size=100, shuffle=True))

    
    # texts = clip_model.labels
    # text_features = clu.get_text_features_list(texts, clip_model.model, device).float()
    with torch.no_grad():
        for index, batch in enumerate(data_loader):

            image, text, label = batch
            image, text, label = image.to(device), text.to(device), label.to(device)

            feature = list_feature_dataloader[index][0].to(device)

            image_features = clip_model.model.encode_image(image).float()
            # image_features_attn, _, _ = clip_model.MoE(x=feature,x2=image_features, train_gate=False)
            domain_feature, _, _ = clip_model.MoE(x=feature,x2=feature, train_gate=False)
            mean_domain_feature = domain_feature.mean(dim=0, keepdim=True)
            _mean_domain_features = mean_domain_feature.repeat_interleave(len(clip_model.labels), dim=0)
            text_features = clip_model._get_text_features(_mean_domain_features.half())
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).float()
            # image_features = torch.mul(
            #     image_features_attn, image_features).detach()
            similarity = clu.get_similarity(image_features, text_features)

            _, indices = similarity.topk(1)
            total += len(label)
            pred = torch.squeeze(indices)
            res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
            res = res.cpu().numpy()
            correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])

        return total, correct
