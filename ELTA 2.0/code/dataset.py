import os
from sklearn.metrics import mean_absolute_error
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class AVADataset(Dataset):
    def __init__(self, csv_path, dataset_path,self_training=False,mode='train',loss_type='emd'):
        self.df = pd.read_csv(csv_path)
        self.dataset_path = dataset_path
        self.mode=mode
        self.loss_type=loss_type

        if mode=='train' and self_training:
            best_beta,best_threshold=self.grid_search()
            print(best_beta,best_threshold)
            _,picked=self.pick_pseudo_labels(best_beta,best_threshold)
            self.df=pd.concat([self.df,picked],ignore_index=True)
        else:
            pass

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])


        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((256, 256)),
                transforms.ToTensor(),
                normalize])
        else:
            self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalize])


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        try:
            image_id=f"{int(row['image_id'])}.jpg"  # AVA
        except(ValueError):
            image_id=row['image_id']    # others

        try:
            image_path = os.path.join(self.dataset_path, image_id)
            image = default_loader(image_path)
        except: 
            image_path = os.path.join('/home/llm/elta/code/diffusion/result/TAD66K/generated', 
                                      row['category'],
                                      image_id)
            image = default_loader(image_path)
            
        if self.loss_type=='emd':
            try:
                scores_names = [f'score{i}' for i in range(2, 12)]
                scores = np.array([row[k] for k in scores_names])
                scores = scores / scores.sum()
            except(KeyError):
                if self.mode=='test':
                    scores = np.array([0.0]*10)
                else:
                    raise KeyError("score2")
        else:
            scores=row['score']/10  # scale to Sigmoid range
        
        image = self.transform(image)
        return image, np.array(scores).astype('float32')


    def softmax(self,x):
        e_x = np.exp(x - np.max(x)) 
        return e_x / e_x.sum()
    
    def temperature_sharping(self, logits_df, mos_list, beta=1.0): 
        mos_mean=np.mean(mos_list)
        factor = np.exp(beta * np.abs(mos_mean - mos_list))
        factor = factor[:, np.newaxis]  # broadcasting
        logits_df*=factor
        sharped_prob=logits_df.apply(self.softmax,axis=1)
        return sharped_prob
    
    def grid_search(self):
        # set the search range
        # beta_choices=[1.0,1.1,1.2,1.25]
        # threshold_choices=[0.70,0.75,0.80,0.85,0.90]
        beta_choices=[1.25]
        threshold_choices=[0.75]

        best_mae=1e6
        for beta in beta_choices:
            for threshold in threshold_choices:
                mae,picked=self.pick_pseudo_labels(beta,threshold)
                print(beta,threshold,mae)
                # if len(picked) > 2000 and mae<best_mae:
                if len(picked) > 10 and mae<best_mae:
                    best_mae=mae
                    best_beta,best_threshold=beta,threshold
                    print('update',best_beta,best_threshold)
        return best_beta,best_threshold


    def pick_pseudo_labels(self,beta=1.0,threshold=0.80):
        '''
        根据logits和score，选出高置信度的伪标签
        '''
        logits = pd.read_csv('logits.csv')
        labels = pd.read_csv('test_result.csv')
        print(f'mae:{mean_absolute_error(labels["gt"].tolist(), labels["score"].tolist())} -> ',end='')
        tmp=logits.drop(columns=['gt','score','image_id']) # dataframe(N*10)
        score=logits['score'].values
        tmp=self.temperature_sharping(tmp,score,beta=beta)
        tmp['max_sum'] = tmp.apply(lambda row: sum(sorted(row, reverse=True)[:1]), axis=1)
        picked_rows = tmp[tmp['max_sum'] > threshold]
        picked=labels.iloc[np.array(picked_rows.index)]
        assert picked.shape[0]>0
        mae=mean_absolute_error(picked['gt'].tolist(), picked['score'].tolist())
        print(f'mae:{mae}',picked.shape[0])
        return mae,picked
