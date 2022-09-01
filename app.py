import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

YOUR_TOKEN="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"
device = "cpu"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN)
pipe.to(device)

def process :
  return "hello"
  
gr.Interface(fn=process, inputs="text", outputs="text").launch()
