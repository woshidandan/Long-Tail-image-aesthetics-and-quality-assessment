import torch.nn as nn
import timm
from einops import rearrange
import torch

class Swin(nn.Module):
    def __init__(self,loss_type='emd'):
        super().__init__()
        self.loss_type=loss_type
        self.vit = timm.create_model('swinv2_base_window8_256', pretrained=True)  # depths=(2, 2, 18, 2)
        self.linear_emd=nn.Sequential(
                nn.Linear(1024, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 10),
                # nn.Softmax(dim=1)
            )
        self.linear_mse = nn.Sequential(
                nn.Dropout(p=0.75),
                nn.Linear(1024,1),
                nn.Sigmoid(),
            )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features = self.vit.forward_features(x) # (B,64,1024)
        features = rearrange(features, 'b (h w) c -> b c h w', h=8, w=8)
        features = self.avgpool(features).flatten(1)
        if self.loss_type=='emd':
            logits=self.linear_emd(features)
            return features,logits,self.softmax(logits)
            # return features,self.linear_emd(features)
        elif self.loss_type=='mse':
            return features,features,torch.squeeze(self.linear_mse(features))
