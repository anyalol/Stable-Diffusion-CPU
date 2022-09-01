import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

YOUR_TOKEN="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"
device = "cpu"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN)
pipe.to(device)


print("hello")