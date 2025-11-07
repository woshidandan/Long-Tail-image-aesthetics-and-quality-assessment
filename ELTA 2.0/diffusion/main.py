import argparse
from torchvision import datasets
from torch.utils.data import Dataset
import torch
from ultra.handler import ModelHandler
from ultra.diffuseMix import DiffuseMix
import random
import os
from PIL import Image

class NestedFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {}
        self.idx_to_class = {}

        for root_dir, _, files in os.walk(root):
            rel_path = os.path.relpath(root_dir, root)
            if rel_path == ".":
                continue
            class_name = rel_path #.replace(os.sep, "_")  # e.g., CAT/Indoor -> CAT_Indoor
            if class_name not in self.class_to_idx:
                idx = len(self.class_to_idx)
                self.class_to_idx[class_name] = idx
                self.idx_to_class[idx] = class_name
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(root_dir, f)
                    self.samples.append((path, self.class_to_idx[class_name]))  # 存储为元组

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# 快速尝试效果
background_prompts = ["change background to a classroom", "snowy","change background to the beach","change background to mountains"]
foreground_prompts = ["add a cat","add a rainbow","add birds","add butterflies","add flowers"]
selected_prompts = background_prompts+foreground_prompts

DATA='TAD66K'  # 数据集名称
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate an augmented dataset from original images and fractal patterns.")
    parser.add_argument('--train_dir', default=f'/home/llm/elta/code/images/{DATA}', type=str, help='Path to the directory containing the original training images.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    model_initialization = ModelHandler(model_id="SD3_UltraEdit_w_mask", device='cuda')
    train_dataset = NestedFolderDataset(root=args.train_dir)
    idx_to_class = train_dataset.idx_to_class
    # train_dataset = datasets.ImageFolder(root=args.train_dir)
    # idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    # all_prompts = [f"change background to {prompt}" for prompt in idx_to_class.values()]
    # selected_prompts = random.sample(all_prompts, min(5, len(all_prompts)))

    augmented_train_dataset = DiffuseMix(
            original_dataset=train_dataset,
            idx_to_class = idx_to_class,
            prompts=selected_prompts,
            model_handler=model_initialization,
            data_name=DATA
    )
        
if __name__ == '__main__':
    main()
    


