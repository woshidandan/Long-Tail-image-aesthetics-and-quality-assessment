import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr


def get_metrics(preds:list,labels:list):
    preds=np.array([pred for pred in preds])
    labels=np.array([label for label in labels])
    if preds.shape!=labels.shape:
        preds=np.squeeze(preds,axis=1)
    plcc = pearsonr(preds, labels)[0]
    srcc = spearmanr(preds, labels)[0]
    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    print(f'plcc:{plcc:.4f}, srcc:{srcc:.4f}, mae:{mae:.4f}, mse:{mse:.4f}')
    return plcc,srcc,mae,mse


def piecewise_mae(df):
    gt = df['gt'].tolist()
    perc = [0, 20, 80, 100]  # 0~20%:low, 20%~80%:medium, 80%~100%:high
    segs = len(perc)
    gt_score = np.percentile(gt, perc)
    for i in range(segs - 1):
        print(f'segment{i}', end=',')
        print(f'[{gt_score[i]},{gt_score[i + 1]}]', end=',')
        tmp = df[(df['gt'] > gt_score[i]) & (df['gt'] < gt_score[i + 1])]
        mae = mean_absolute_error(tmp['gt'].tolist(), tmp['score'].tolist())
        mse = mean_squared_error(tmp['gt'].tolist(), tmp['score'].tolist())
        print(f'mae:{mae},mse:{mse}')



if __name__ == "__main__":
    csv_list=["/home/llm/codebase/LTIAA/results/ours_ava.csv"]
    for csv_name in csv_list:
        print('-'*50)
        print(csv_name)
        df=pd.read_csv(csv_name)
        get_metrics(df['gt'].tolist(),df['score'].tolist())
        piecewise_mae(df)

