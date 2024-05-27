import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from torch import nn

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, targets, preds):
        assert targets.shape == preds.shape
        cdf_target = torch.cumsum(targets, dim=1)
        cdf_estimate = torch.cumsum(preds, dim=1)

        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        return samplewise_emd.mean()


def get_score(output):
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor)
    w = w.to(torch.device("cuda"))
    w = w.repeat(output.size(0), 1)
    score = (output * w).sum(dim=1)
    return score.data.cpu().numpy().tolist()


def get_flops_params(model, inputs):
    print('=' * 50)
    from thop import profile
    flops, params = profile(model, inputs=(inputs,))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))


def tSNE(features, labels):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    features = features.to_numpy()
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(features)
    x_min, x_max = np.min(result, 0), np.max(result, 0)
    result = (result - x_min) / (x_max - x_min)
    fig = plt.figure()
    
    colors=['aqua','lime','gold','violet','cornflowerblue','lightcoral','chocolate','palegreen','yellow','deepskyblue']
            
    df = pd.DataFrame({'x': result[:,0], 'y': result[:,1],'z':labels})
    df.to_csv('toy_exps/scatter_iaa.csv',index=None,sep=',')
    ax = fig.add_subplot(111)
    for _,row in df.iterrows():
        ax.scatter(row['x'],row['y'],c=colors[int(row['z'])],marker='o')

    plt.savefig('./plot/tsne_iaa.jpg')


def output_evaluation_results(testset_path, preds, features,logits,outputs):
    df = pd.read_csv(testset_path)
    ids = df['image_id'].tolist()
    labels = df['score'].to_numpy()

    # df = pd.DataFrame({'image_id': ids, 'gt': labels, 'score': preds})
    # df.to_csv(f'test_result.csv', index=None, sep=',')
    columns_names = ['score' + str(i) for i in range(2, 12)]
    outputs.columns = columns_names
    logits.columns = columns_names
    df = pd.DataFrame({'image_id': ids, 'gt': labels, 'score': preds})
    df_1 = pd.concat([df,outputs],axis=1)
    df_1.to_csv(f'test_result.csv', index=None, sep=',')
    df_2 = pd.concat([df,logits],axis=1)
    df_2.to_csv(f'logits.csv', index=None, sep=',')
    # # torch.save(features,'features.pth')
    # tSNE(features, labels)


def simloss(features, targets):
    x = features
    y = torch.Tensor(get_score(targets)).unsqueeze(1) # distributed
    # y=targets*torch.ones((features.shape[0],features.shape[0]),device='cuda:0')    # single-valued
    feat_sim_mat = torch.matmul(F.normalize(x.view(x.size(0),-1)), F.normalize(x.view(x.size(0),-1)).permute(1,0))
    ratio=(torch.abs(y-y.t()))/(torch.max(y)-torch.min(y))
    # label_sim_mat = 1-hyper1*(ratio**hyper2)
    label_sim_mat = 1-1*(ratio**1)
    label_sim_mat = label_sim_mat.to(torch.device("cuda"))
    return F.mse_loss(feat_sim_mat, label_sim_mat)