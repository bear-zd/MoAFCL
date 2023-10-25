import torch
import numpy as np
import utils.cliputils as clu 
from model.clip import ClipModelMA, Adapter
from torch.utils.data import DataLoader
import torch.nn as nn



def test_client(clip_model: ClipModelMA, adapter ,data_loader: DataLoader, device):
    clip_model.model.eval()
    adapter.eval()
    total = 0
    correct = 0
    texts = clip_model.labels
    text_features = clu.get_text_features_list(texts, clip_model.model, device).float()

    with torch.no_grad():
        for batch in data_loader:

            image, text, label = batch
            image, text, label = image.to(device), text.to(device), label.to(device)
            image_features = clip_model.model.encode_image(image).float()
            image_features_att = adapter(image_features)
            image_features = torch.mul(
                image_features_att, image_features).detach()
            similarity = clu.get_similarity(image_features, text_features)

            _, indices = similarity.topk(1)
            total += len(label)
            pred = torch.squeeze(indices)
            res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
            res = res.cpu().numpy()
            correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])

        return correct/total



def train_client(clip_model : ClipModelMA, image_adapter: Adapter, dataloader, optimizer, device):
    clip_model.model.train()
    image_adapter.train()
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    for batch in dataloader:
        image, text, label = batch

        image = image.to(device)
        text = text.to(device)

        image_features = clip_model.model.encode_image(image).float()
        text_features = clip_model.model.encode_text(text).float()
        image_features_att = image_adapter(image_features)
        image_features = torch.mul(image_features_att, image_features)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = clip_model.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        ground_truth = torch.arange(len(image), dtype=torch.long, device=device)

        loss = (loss_img(logits_per_image, ground_truth) + 
                loss_txt(logits_per_text, ground_truth))/2


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return image_adapter

