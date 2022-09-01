import gradio as gr
import torch
from torch import autocast
from PIL import Image
from diffusers import StableDiffusionPipeline

print("hello sylvain")

YOUR_TOKEN="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"
device="cpu"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN)
pipe.to(device)

def infer(prompt):
    image = pipe(prompt)["sample"][0]
    return image
  
gr.Interface(fn=infer, inputs="text", outputs="image").launch()

print("Great sylvain ! Everything is working fine !")