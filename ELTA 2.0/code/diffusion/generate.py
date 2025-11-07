# For Editing with SD3
from pathlib import Path
import torch
from diffusers import StableDiffusion3InstructPix2PixPipeline
from diffusers.utils import load_image
import requests
import PIL.Image
import PIL.ImageOps

# 创建保存结果的目录
save_dir = Path("result/ultra")
save_dir.mkdir(parents=True, exist_ok=True)

# 定义多个prompts
prompts = [
    "apse_indoor",
    "badlands",
    "bathroom",
    "bus_interior",
    "campus",
    "change background to bus_interior",
    "change background to campus",
    "change background to badlands",
    "change background to apse_indoor",
    "change background to bathroom"
]

pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained("/home/llm/elta/code/diffusion/SD3_UltraEdit_w_mask", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

img = load_image("/home/llm/elta/code/images/TAD66K/apse/indoor/131316467@N0623928254427.jpg").resize((512, 512))
# For free form Editing, seed a blank mask
mask_img = PIL.Image.new("RGB", img.size, (255, 255, 255))
# 处理每个prompt
for i, prompt in enumerate(prompts):
    image = pipe(
        prompt,
        image=img,
        mask_img=mask_img,
        negative_prompt="",
        num_inference_steps=50,
        image_guidance_scale=1.5,
        guidance_scale=6.0,
    ).images[0]
    
    # 保存结果图像
    save_path = save_dir / f"edited_image_4_{i+1}_{prompt.replace(' ','_')}.png"
    image.save(save_path)
    print(f"Saved: {save_path}")
# display images   python generate.py