import utils.cliputils as clu
from utils.dataload import *
from model.clip import ClipModelMA
import torch.optim as optim
from tqdm import tqdm
from torch.nn.functional import binary_cross_entropy_with_logits
import torch

def train_server(clip_model: ClipModelMA, data, index, device):
    clip_model.MoE = clip_model.MoE.to(device)
    optimizer = optim.SGD(clip_model.MoE.parameters(), lr=0.001, momentum=0.9)
    clip_model.MoE.train()
    logging.info(f"Server start to train MoE !")
    for epoch in tqdm(range(5)):
        for batch in data:
            image, _, _ = batch

            image = image.to(device)

            image_features = clip_model.model.encode_image(image).float()
            _ , loss_gate, logits = clip_model.MoE(image_features)

            one_hot_label = torch.ones(logits.shape, device=device)*0.2
            one_hot_label[0][index] = 1
            
            loss_label = binary_cross_entropy_with_logits(
                logits, one_hot_label)

            loss = loss_gate + loss_label
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()






def test_server(clip_model: ClipModelMA, data_loader: DataLoader, device):
    clip_model.model.eval()
    clip_model.MoE.eval()
    total = 0
    correct = 0
    texts = clip_model.labels
    text_features = clu.get_text_features_list(texts, clip_model.model, device).float()
    with torch.no_grad():
        for batch in data_loader:

            image, text , label = batch
            image, text, label = image.to(device), text.to(device), label.to(device)

            image_features = clip_model.model.encode_image(image).float()
            image_features_attn, _, _ = clip_model.MoE(image_features)
            image_features = torch.mul(
                image_features_attn, image_features).detach()
            similarity = clu.get_similarity(image_features, text_features)

            _, indices = similarity.topk(1)
            total += len(label)
            pred = torch.squeeze(indices)
            res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
            res = res.cpu().numpy()
            correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])

        return correct/total
