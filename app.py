import gradio as gr
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

print("hello sylvain")

YOUR_TOKEN="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"

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN)
pipe.to("cpu")

def process(name):
  return "hello " + name
  
gr.Interface(fn=process, inputs="text", outputs="text").launch()
