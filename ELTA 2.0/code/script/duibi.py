#python script/duibi.py

import os
import matplotlib.pyplot as plt
import glob
from PIL import Image
import re

# 定义路径
original_path = "/home/llm/elta/code/images/TAD66K/art_studio"
generated_path = "/home/llm/elta/code/diffusion/result/generated/art_studio"

def plot_comparison(original_img_path):
    # 获取原始图片名
    img_name = os.path.basename(original_img_path)
    
    # 查找对应的生成图片
    generated_files = glob.glob(os.path.join(generated_path, f"{img_name}_generated_*.jpg"))
    
    # 限制生成图片数量为10个
    generated_files = generated_files[:10]

    # 创建子图
    fig = plt.figure(figsize=(25, 10))  # 调整图片大小以适应新布局
    
    # 显示原始图片
    ax1 = plt.subplot(3, 4, 1)
    img = Image.open(original_img_path)
    ax1.imshow(img)
    ax1.set_title('Original')
    ax1.axis('off')
    
    # 显示生成的图片
    for idx, gen_path in enumerate(generated_files, start=2):
        ax = plt.subplot(3, 4, idx)
        img = Image.open(gen_path)
        ax.imshow(img)

        # 从文件名中提取提示词
        prompt_match = re.search(r'generated_(.*?)_\d+\.jpg$', os.path.basename(gen_path))
        prompt = prompt_match.group(1) if prompt_match else ''
        
        ax.set_title(f'{prompt}', fontsize=8)  # 显示提示词
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # 获取所有原始图片
    original_images = glob.glob(os.path.join(original_path, "*.jpg"))
    
    # 为每个原始图片创建对比图
    for orig_img in original_images:
        fig = plot_comparison(orig_img)
        
        # 保存图片
        save_name = f"comparison_{os.path.basename(orig_img)}.png"
        fig.savefig(save_name)
        plt.close(fig)
        print(f"Saved comparison for {os.path.basename(orig_img)}")

if __name__ == "__main__":
    main()