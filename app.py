import gradio as gr
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

YOUR_TOKEN="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"

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN)
pipe.to("cuda")

prompt = "a photograph of an astronaut riding a horse"

image = pipe(prompt)["sample"][0]

result = pipe(prompt)
print(result)