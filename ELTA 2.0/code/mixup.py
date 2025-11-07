import torch
import numpy as np
from utils import get_score

np.set_printoptions(suppress=True)

def get_mixup_pairs(labels,tau_1=0.2,tau_2=5,mixup_pair_numbers=1): 
    batch_size = labels.size()[0]  
    mos_list = get_score(labels) if labels.dim()==2 else labels.tolist()
    mos_mean = np.mean(mos_list)
    # P(i)=\frac{\exp{(|\bar{s}-s_i|/ \tau_1)}}{\sum_{k=1}^B\exp{(|\bar{s}-s_k|/ \tau_1)}}
    prob_list = [np.exp((np.abs(mos_mean-i))/tau_1) for i in mos_list] 
    prob_list = prob_list / np.sum(prob_list)
    # The indexes of first image  
    select_indexes = np.random.choice(list(range(batch_size)),size=mixup_pair_numbers, p=prob_list,replace=False)
    # The indexes of second image
    pair_indexes=[]
    for index in select_indexes:
        index_score=mos_list[index] 
        # P(j|i)=\frac{\exp{(\tau_2 /|s_i-s_j|)}}{\sum_{k=1,2,\cdots,i-1,i+1,\cdots,B}\exp{(\tau_2/|s_i-s_k|)}}
        prob_list_pair=[np.exp(tau_2/(np.abs(index_score-i)+1e-2)) for i in mos_list]
        prob_list_pair[index]=0
        prob_list_pair = prob_list_pair / np.sum(prob_list_pair)
        pair_indexes.append(np.random.choice(list(range(batch_size)), p=prob_list_pair))

    mixup_factors=[]
    for i in range(len(select_indexes)):
        p1=prob_list[select_indexes[i]]
        p2=prob_list[pair_indexes[i]]
        mixup_factors.append(p1/(p1+p2))    # λ in the paper

    return select_indexes,pair_indexes,mixup_factors


def mixup(features, targets,tau_1,tau_2):
    index_1, index_2, alpha = get_mixup_pairs(targets,tau_1,tau_2)
    # if len(index_1)==0:
    #     return features,targets
    features_1, features_2 = features[index_1, ...], features[index_2, ...]
    labels_1, labels_2 = targets[index_1, ...], targets[index_2, ...]
    mixed_features = torch.zeros_like(features_1)
    mixed_labels = torch.zeros_like(labels_1)
    for i, s in enumerate(alpha):
        mixed_features[i, ...] = features_1[i, ...] * s + features_2[i, ...] * (1 - s)
        mixed_labels[i, ...] = labels_1[i, ...] * s + labels_2[i, ...] * (1 - s)
    return torch.cat((features,mixed_features),dim=0), torch.cat((targets,mixed_labels),dim=0)
