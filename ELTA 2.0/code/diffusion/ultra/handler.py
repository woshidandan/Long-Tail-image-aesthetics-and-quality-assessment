import torch
from diffusers import StableDiffusion3InstructPix2PixPipeline
from diffusers.utils import load_image
import PIL.Image


class ModelHandler:
    def __init__(self, model_id, device):
        self.pipeline = StableDiffusion3InstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipeline = self.pipeline.to(device)
        # print(self.pipeline)
    def generate_images(self, prompt, img_path,num_inference_steps,image_guidance_scale, guidance_scale,negative_prompt=""):
        # img = PIL.Image.open(img_path).convert('RGB')
        img = load_image(img_path).resize((512, 512))
        mask_img = PIL.Image.new("RGB", img.size, (255, 255, 255))
        return self.pipeline(prompt, image=img, mask_img=mask_img, negative_prompt=negative_prompt,num_inference_steps=num_inference_steps,image_guidance_scale=image_guidance_scale,guidance_scale=guidance_scale).images
