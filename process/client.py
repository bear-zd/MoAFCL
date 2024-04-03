import torch
import numpy as np
import utils.cliputils as clu 
from model.clip import ClipModelMA, Adapter, Client
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


@torch.no_grad()
def test_client(clip_model: ClipModelMA, client: Client ,data_loader: DataLoader, device):
    clip_model.model.eval()
    if client is not None:
        client.adapter.eval()
    total = 0
    correct = 0
    # texts = clip_model.labels
    # text_features = clu.get_text_features_list(texts, clip_model.model, device).float()
    client_domain_feature = torch.tensor(client.preprocess(), dtype=torch.float)
    list_image_domain_features = list(DataLoader(TensorDataset(client_domain_feature),batch_size=100, shuffle=False))
    # domain_feature = np.stack([client.adapter(image_feature) for image_feature in image_features])
    # dataset = TensorDataset(domain_feature)
    # list_domain_dataloader = list(DataLoader(dataset=dataset, batch_size=100, shuffle=True))
    # mean_domain_features = [feature[0].mean(dim=0, keepdim=True) for feature in list_domain_dataloader]
    # _mean_domain_features = [feature.repeat_interleave(len(clip_model.labels), dim=0) for feature in mean_domain_features]
    # text_features = torch.cat([clip_model._get_text_features(feature) for feature in _mean_domain_features])

    with torch.no_grad():
        for index, batch in enumerate(data_loader):
            domain_feature = client.adapter(list_image_domain_features[index][0].to(device))
            mean_domain_feature = domain_feature.mean(dim=0, keepdim=True)
            _mean_domain_features = mean_domain_feature.repeat_interleave(len(clip_model.labels), dim=0)
            text_features = clip_model._get_text_features(_mean_domain_features.half())


            image, text, label = batch
            image, text, label = image.to(device), text.to(device), label.to(device)
            image_features = clip_model.model.encode_image(image).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).float()

            # if client is not None:
            #     image_features_att = client.adapter(image_features)
            #     image_features = torch.mul(
            #         image_features_att, image_features)
            similarity = clu.get_similarity(image_features, text_features)

            _, indices = similarity.topk(1)
            total += len(label)
            pred = torch.squeeze(indices)
            res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
            res = res.cpu().numpy()
            correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])

        return correct/total



def train_client(clip_model : ClipModelMA, client: Client, dataloader, device, args):
    optimizer = optim.Adam(params=[{"params":client.adapter.parameters()}], lr=args.lr, betas=(
                    args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay) 
    clip_model.model.to(device)
    client.adapter.to(device)
    clip_model.model.train()
    client.adapter.train()
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    for image, _, _ in dataloader:
        image = image.to(device)
        _ = clip_model.model.encode_image(image).float() # just for the hook work
    client.temp_hook.remove()
    client_domain_feature = torch.tensor(client.preprocess(), dtype=torch.float)
    # print(client_domain_feature.shape) # 1000(total number) * 768(feature num)
    list_image_domain_features = list(DataLoader(TensorDataset(client_domain_feature), batch_size=100, shuffle=False))
    for index, batch in enumerate(dataloader):
        image, _, label = batch
        label = label.to(device)
        image = image.to(device)
        # text = text.to(device)
        domain_feature = client.adapter(list_image_domain_features[index][0].to(device))
        mean_domain_feature = domain_feature.mean(dim=0, keepdim=True)
        _mean_domain_features = mean_domain_feature.repeat_interleave(len(clip_model.labels), dim=0)
    
        text_features = clip_model._get_text_features(_mean_domain_features.half())
        
        
        image_features = clip_model.model.encode_image(image).float()

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = clip_model.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features.half() @ text_features.t()
        # logits_per_text = logits_per_image.t()
        # print(torch.tensor(label))
        # ground_truth = torch.arange(len(image), dtype=torch.long, device=device)
        
        loss = F.cross_entropy(logits_per_image, label)
        # loss = (loss_img(logits_per_image, ground_truth))
                # + loss_txt(logits_per_text, ground_truth))/2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

