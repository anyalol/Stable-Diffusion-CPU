import os
MY_SECRET_TOKEN=os.environ.get('HF_TOKEN_SD')
os.system("git clone https://github.com/huggingface/diffusers")
import sys
sys.path.append("diffusers")
import gradio as gr
import torch
#from torch import autocast // only for GPU
from PIL import Image
#from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline

print("hello sylvain")

YOUR_TOKEN=MY_SECRET_TOKEN

device="cpu"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN)
pipe.to(device)

def infer(prompt, init_image):
    
    image = pipe(prompt, init_image=init_image)["sample"][0]
    
    return image

print("Great sylvain ! Everything is working fine !")

title="Stable Diffusion CPU"
description="Stable Diffusion example using CPU and HF token. Warning: Slow process... ~5/10 min inference time" 

gr.Interface(fn=infer, inputs=["text", "image"], outputs="image",title=title,description=description).launch(enable_queue=True)

