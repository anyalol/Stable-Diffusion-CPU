import os

import gradio as gr
import torch
from torch import autocast
from PIL import Image
from diffusers import StableDiffusionPipeline

print("hello sylvain")

YOUR_TOKEN=os.environ.getattribute("HF_TOKEN_SD")

device="cpu"

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN)
pipe.to(device)

def infer(prompt):
    image = pipe(prompt)["sample"][0]
    return image

print("Great sylvain ! Everything is working fine !")

title="Stable Diffusion CPU"
description="Stable Diffusion example using CPU and HF token. Warning: Slow process..." 

gr.Interface(fn=infer, inputs="text", outputs="image",title=title,description=description).launch(enable_queue=True)

