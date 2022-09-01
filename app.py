import gradio as gr

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
from PIL import Image  
import re

YOUR_TOKEN="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"

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=YOUR_TOKEN)

prompt = "a photograph of an astronaut riding a horse"

image = pipe(prompt)["sample"][0]

result = pipe(prompt)
print(result)