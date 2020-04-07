import torch
import torch.nn as nn
import numpy as np

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, anchor, pos, neg, alpha):
        pos_dist = torch.sum(torch.sub(anchor, pos).pow(2), 1)
        neg_dist = torch.sum(torch.sub(anchor, neg).pow(2), 1)
        loss = torch.add(torch.sub(pos_dist, neg_dist), alpha)
        mean_loss = torch.mean(torch.clamp_min(loss, 0.0), 0)
        return mean_loss

def triplet_loss(anchor, pos, neg, alpha):
    """
    :param anchor:
    :param pos:
    :param neg:
    :param alpha:
    :return:
    """
    pos_dist = torch.sum(torch.sub(anchor, pos).pow(2), 1)
    neg_dist = torch.sum(torch.sub(anchor, neg).pow(2), 1)

    loss = torch.add(torch.sub(pos_dist, neg_dist), alpha)
    mean_loss = torch.mean(torch.clamp_min(loss, 0.0), 0)

    return mean_loss


def sample_batch(dateset, people_per_batch, images_per_people):
    pass


def select_triplet(embedding, labels, alpha = 0.2):
    anchor = []
    pos = []
    neg = []
    for i in range(len(embedding)):#对第i个图片
        neg_dis = np.sum(np.square(embedding[i] - embedding), 1)
        for p in range(len(embedding)):
            if labels[i] == labels[p]:
                neg_dis[p] = np.NaN
        for j in range(i+1, len(embedding)):
            if labels[i] == labels[j]:
                pos_dis = np.sum(np.square((embedding[i]-embedding[j])))
                all_neg = np.where(neg_dis - pos_dis < alpha)[0]
                all_neg_num = all_neg.shape[0]
                if all_neg_num > 0:
                    neg_idx = all_neg[np.random.randint(all_neg_num)]
                    anchor.append(i)
                    pos.append(j)
                    neg.append(neg_idx)

    return anchor, pos, neg
