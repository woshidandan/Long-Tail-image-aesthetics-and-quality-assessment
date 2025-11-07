import os
from torch.utils.data import Dataset
from PIL import Image
import random
from augment.utils import Utils


class DiffuseMix(Dataset):
    def __init__(self, original_dataset, num_images, guidance_scale, idx_to_class, prompts, model_handler,data_name='TAD66K'):
        self.original_dataset = original_dataset
        self.idx_to_class = idx_to_class
        self.combine_counter = 0
        self.prompts = prompts
        self.model_handler = model_handler
        self.num_augmented_images_per_image = num_images
        self.guidance_scale = guidance_scale
        # self.data_name = data_name
        self.utils = Utils()
        self.augmented_images = self.generate_augmented_images()

    def generate_augmented_images(self):
        augmented_data = []

        base_directory = './result'
        # base_directory = os.path.join('./result', self.data_name)
        # original_resized_dir = os.path.join(base_directory, 'original_resized')
        generated_dir = os.path.join(base_directory, 'generated')

        # Ensure these directories exist
        # os.makedirs(original_resized_dir, exist_ok=True)
        os.makedirs(generated_dir, exist_ok=True)

        for idx, (img_path, label_idx) in enumerate(self.original_dataset.samples):

            label = self.idx_to_class[label_idx]  # Use folder name as label

            original_img = Image.open(img_path).convert('RGB')
            original_img = original_img.resize((256, 256))
            img_filename = os.path.basename(img_path)

            # label_dirs = {dtype: os.path.join(base_directory, dtype, str(label)) for dtype in
            #               ['original_resized', 'generated']}
            label_dirs = {dtype: os.path.join(base_directory, dtype, str(label)) for dtype in
                          ['generated']}

            for dir_path in label_dirs.values():
                os.makedirs(dir_path, exist_ok=True)

            # original_img.save(os.path.join(label_dirs['original_resized'], img_filename))

            for prompt in self.prompts:
                augmented_images =  self.model_handler.generate_images(prompt, img_path, self.num_augmented_images_per_image,
                                                          self.guidance_scale)

                for i, img in enumerate(augmented_images):
                    img = img.resize((256, 256))
                    generated_img_filename = f"{img_filename}_generated_{prompt}_{i}.jpg"
                    img.save(os.path.join(label_dirs['generated'], generated_img_filename))
                    augmented_data.append((img, label))

        return augmented_data

    def __len__(self):
        return len(self.augmented_images)

    def __getitem__(self, idx):
        image, label = self.augmented_images[idx]
        return image, label
