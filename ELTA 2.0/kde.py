import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import os
import shutil

DATA = 'TAD66K'  # 数据集名称
def kde():
    '''
    kde() 函数：从原始标注数据中根据“评分稀有性”选择图像样本
    TODO: 1、这里的alpha会影响抽样，需要修改
    2、生成模型的prompts可以修改
    3、生成模型本身可以替换为别的
    '''         # 数据集名称
    alpha = 10.0               # 稀有性指数放大参数，数字越大代表对尾部的偏好越强（非常重要，需谨慎调整）
    samples_per_class = 10
    input_csv=f'dataset_backup/{DATA}/train_select.csv'       # 处理好的同时包含图像美感评分和类别信息的标注文件
    output_csv=f'dataset_backup/{DATA}/selected.csv'   # 被抽样选中用于数据增强的图像信息存在这个目录下

    df = pd.read_csv(input_csv)
    selected_rows = []
    for category_name, group_df in df.groupby('category'):
        scores = group_df['score'].values
        
        if len(scores) <= samples_per_class:
            chosen_df = group_df
        else:
            kde = gaussian_kde(scores)
            density_estimates = kde(scores)
            density_estimates = np.clip(density_estimates, a_min=1e-8, a_max=None)
            
            rarity_scores = 1.0 / density_estimates
            sampling_prob = rarity_scores ** alpha
            sampling_prob /= sampling_prob.sum()
            
            chosen_indices = np.random.choice(len(scores), size=samples_per_class, replace=False, p=sampling_prob)
            chosen_df = group_df.iloc[chosen_indices]
        
        selected_rows.append(chosen_df)

    final_selected_df = pd.concat(selected_rows, ignore_index=True)

    final_selected_df.to_csv(output_csv, index=False)

def prepare_images():
    '''
    prepare_images() 函数：将抽样选中的图像从原始数据集中复制到新的目录中
    '''
    csv_path=f'dataset_backup/{DATA}/selected.csv'
    source_dir=f'/data/dataset/{DATA}'

    df=pd.read_csv(csv_path)

    for category_name, group_df in df.groupby('category'):
        image_ids=group_df['image_id'].tolist()
        target_dir=f'images/{DATA}/'+category_name
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for image_id in image_ids:
            source_path = os.path.join(source_dir, image_id)
            target_path = os.path.join(target_dir, image_id)
            
            if os.path.isfile(source_path):
                shutil.copy(source_path, target_path)

if __name__=='__main__':
    kde()
    prepare_images()