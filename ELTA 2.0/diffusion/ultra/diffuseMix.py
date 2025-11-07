import os
import re
from torch.utils.data import Dataset
from PIL import Image
import random


class DiffuseMix(Dataset):
    def __init__(self, original_dataset,idx_to_class, prompts, model_handler,data_name):
        self.original_dataset = original_dataset
        self.idx_to_class = idx_to_class
        self.prompts = prompts
        self.model_handler = model_handler
        self.guidance_scale = 6.0
        self.negative_prompt=""
        self.num_inference_steps=50
        self.image_guidance_scale=1.5
        self.data_name = data_name
        self.augmented_images = self.generate_augmented_images()

    def generate_augmented_images(self):
        augmented_data = []
        base_directory = os.path.join('./result', self.data_name)
        generated_dir = os.path.join(base_directory, 'generated')
        os.makedirs(generated_dir, exist_ok=True)

        for idx, (img_path, label_idx) in enumerate(self.original_dataset.samples):

            label = self.idx_to_class[label_idx]  # Use folder name as label

            img_filename = os.path.basename(img_path)

            label_dirs = {dtype: os.path.join(base_directory, dtype, str(label)) for dtype in
                          ['generated']}

            for dir_path in label_dirs.values():
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created directory: {dir_path}")

            # all_prompts = [f"change background to {re.sub(r'[\\/]+', ' ', prompt)}" for prompt in self.idx_to_class.values()]
            all_prompts = []
            for prompt in self.idx_to_class.values():
                clean_prompt = re.sub(r'[\\/]+', ' ', prompt)
                all_prompts.append(f"change background to {clean_prompt}")
            selected_prompts = random.sample(all_prompts, min(5, len(all_prompts)))

            for prompt in selected_prompts:                
                augmented_images =  self.model_handler.generate_images(prompt, img_path,negative_prompt = self.negative_prompt,num_inference_steps = self.num_inference_steps,image_guidance_scale = self.image_guidance_scale,guidance_scale = self.guidance_scale)

                for i, img in enumerate(augmented_images):
                    generated_img_filename = f"{img_filename}_generated_{prompt}_{i}.jpg"
                    img.save(os.path.join(label_dirs['generated'], generated_img_filename))
                    augmented_data.append((img, label))

            # for prompt in self.prompts:                
            #     augmented_images =  self.model_handler.generate_images(prompt, img_path,negative_prompt = self.negative_prompt,num_inference_steps = self.num_inference_steps,image_guidance_scale = self.image_guidance_scale,guidance_scale = self.guidance_scale)

            #     for i, img in enumerate(augmented_images):
            #         # img = img.resize((256, 256))
            #         generated_img_filename = f"{img_filename}_generated_{prompt}_{i}.jpg"
            #         img.save(os.path.join(label_dirs['generated'], generated_img_filename))
            #         augmented_data.append((img, label))

        return augmented_data

    def __len__(self):
        return len(self.augmented_images)

    def __getitem__(self, idx):
        image, label = self.augmented_images[idx]
        return image, label
