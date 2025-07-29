# Long-Tail-image-aesthetics-and-quality-assessment
ICML2024: 首篇针对IAA中的长尾问题提出解决方案的工作


[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)


<div align="center">
<h1>
<b>
ELTA: An Enhancer against Long-Tail for Aesthetics-oriented Models
</b>
</h1>
<h4>
<b>
Limin Liu*, Shuai He*, Anlong Ming*, Rui Xie, Huadong Ma
    
Beijing University of Posts and Telecommunications, *Equal contribution
</b>
</h4>
</div>

-----------------------------------------



## Introduction
现实世界中的数据集往往呈现长尾分布，影响了模型的泛化能力和公平性。这个问题在图像美学评估（IAA）任务中尤为突出，由于特征和标签之间存在严重的分布不匹配，以及美学对图像变化的高度敏感性，这种不平衡很难得到缓解。
为了解决这些问题，我们提出了一种面向美学模型的长尾增强器（ELTA）。ELTA 首先利用专门设计的Mixup技术来增强高层空间中的Minority特征表示，同时保留图像固有的美学品质。然后通过相似度一致性方法对齐特征和标签，有效缓解了分布不匹配问题。最后，ELTA 采用了一个自适应的锐化策略改善模型的输出logits分布，从而提高伪标签的质量。

<img src="pipeline_final.png">

## 环境
* einops==0.4.1
* matplotlib==3.3.4
* nni==2.6.1
* numpy==1.19.5
* pandas==1.1.5
* Pillow==10.2.0
* scikit_learn==1.4.0
* scipy==1.5.4
* timm==0.6.12
* torch==1.10.1
* torchvision==0.11.2
* tqdm==4.64.1









## 模型训练
```
python main.py --csv_path           [dataset annotation file path]
               --dataset_path       [dataset image path]
               --mixup              # optional, enable TFA module
               --simloss_weight 1   # optional, enable FLSA module and specify weight
               ...                  # other arguments
```

权重文件: https://drive.google.com/file/d/1pA7kOCPHEUR5oNnocBZHH41Erud9Y30S/view?usp=drive_link

## 模型推理
```
python main.py -e                   [dataset annotation file path]
               --test_dataset_path  [dataset image path]
               --resume             [checkpoint path]   # required!
               ...                  # other arguments
```

## 在第一轮训练之后启用自训练
```
python main.py --st                 # enable self-training
               ...                  # other arguments
```

## 建议: 使用 NNI 自动调参
```
# 请先修改main_nni.py文件中命令 'trial_command' 和搜索空间 'search_space'
python main_nni.py
```

## Related Work from Our Group
<table>
  <thead align="center">
    <tr>
      <td><b>🎁 Projects</b></td>
      <td><b>📚 Publication</b></td>
      <td><b>🌈 Content</b></td>
      <td><b>⭐ Stars</b></td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://github.com/woshidandan/Attacker-against-image-aesthetics-assessment-model"><b>Attacker Against IAA Model【美学模型的攻击和安全评估框架】</b></a></td>
      <td><b>TIP 2025</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Attacker-against-image-aesthetics-assessment-model?style=flat-square&labelColor=343b41"/></td>
    </tr
    <tr>
      <td><a href="https://github.com/woshidandan/Rethinking-Personalized-Aesthetics-Assessment"><b>Personalized Aesthetics Assessment【个性化美学评估新范式】</b></a></td>
      <td><b>CVPR 2025</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Rethinking-Personalized-Aesthetics-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Pixel-level-No-reference-Image-Exposure-Assessment"><b>Pixel-level image exposure assessment【首个像素级曝光评估】</b></a></td>
      <td><b>NIPS 2024</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Pixel-level-No-reference-Image-Exposure-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Long-Tail-image-aesthetics-and-quality-assessment"><b>Long-tail solution for image aesthetics assessment【美学评估数据不平衡解决方案】</b></a></td>
      <td><b>ICML 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Long-Tail-image-aesthetics-and-quality-assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Prompt-DeT"><b>CLIP-based image aesthetics assessment【基于CLIP多因素色彩美学评估】</b></a></td>
      <td><b>Information Fusion 2024</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Prompt-DeT?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/SR-IAA-image-aesthetics-and-quality-assessment"><b>Compare-based image aesthetics assessment【基于对比学习的多因素美学评估】</b></a></td>
      <td><b>ACMMM 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/SR-IAA-image-aesthetics-and-quality-assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Image-Color-Aesthetics-and-Quality-Assessment"><b>Image color aesthetics assessment【首个色彩美学评估】</b></a></td>
      <td><b>ICCV 2023</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Image-Color-Aesthetics-and-Quality-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Image-Aesthetics-and-Quality-Assessment"><b>Image aesthetics assessment【通用美学评估】</b></a></td>
      <td><b>ACMMM 2023</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Image-Aesthetics-and-Quality-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/TANet-image-aesthetics-and-quality-assessment"><b>Theme-oriented image aesthetics assessment【首个多主题美学评估】</b></a></td>
      <td><b>IJCAI 2022</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/TANet-image-aesthetics-and-quality-assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/AK4Prompts"><b>Select prompt based on image aesthetics assessment【基于美学评估的提示词筛选】</b></a></td>
      <td><b>IJCAI 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/AK4Prompts?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/mRobotit/M2Beats"><b>Motion rhythm synchronization with beats【动作与韵律对齐】</b></a></td>
      <td><b>IJCAI 2024</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/mRobotit/M2Beats?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Champion-Solution-for-CVPR-NTIRE-2024-Quality-Assessment-on-AIGC"><b>Champion Solution for AIGC Image Quality Assessment【NTIRE AIGC图像质量评估赛道冠军】</b></a></td>
      <td><b>CVPRW NTIRE 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Champion-Solution-for-CVPR-NTIRE-2024-Quality-Assessment-on-AIGC?style=flat-square&labelColor=343b41"/></td>
    </tr>
  </tbody>
</table>
