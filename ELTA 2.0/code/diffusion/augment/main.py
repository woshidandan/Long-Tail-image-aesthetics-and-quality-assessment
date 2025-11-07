import argparse
from torchvision import datasets
import torch
from augment.handler import ModelHandler
from augment.diffuseMix import DiffuseMix
import random


# 快速尝试效果
# atmosphere_prompts = ["Add a warm, inviting atmosphere to the image", "a sketch with crayon"]
background_prompts = ["change background to a classroom", "snowy","change background to the beach","change background to mountains"]
foreground_prompts = ["add a cat","add a rainbow","add birds","add butterflies","add flowers"]
selected_prompts = background_prompts+foreground_prompts
# 可以增加更多的风格效果
# all_prompts = ["Autumn", "snowy", "watercolor art","sunset", "rainbow", "aurora",
#                "mosaic", "ukiyo-e", "a sketch with crayon"]

DATA='TAD66K'  # 数据集名称
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate an augmented dataset from original images and fractal patterns.")
    parser.add_argument('--train_dir', default=f'/home/llm/elta/code/images/{DATA}', type=str, help='Path to the directory containing the original training images.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    model_initialization = ModelHandler(model_id="instruct-pix2pix", device='cuda')
    train_dataset = datasets.ImageFolder(root=args.train_dir)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    
    # all_prompts = [f"change background to {prompt}" for prompt in idx_to_class.values()]
    # selected_prompts = random.sample(all_prompts, min(5, len(all_prompts)))

    augmented_train_dataset = DiffuseMix(
            original_dataset=train_dataset,
            num_images=1,
            guidance_scale=6,
            idx_to_class = idx_to_class,
            prompts=selected_prompts,
            model_handler=model_initialization,
            data_name=DATA
    )
        
if __name__ == '__main__':
    main()
    


