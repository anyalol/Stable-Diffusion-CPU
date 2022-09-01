import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

YOUR_TOKEN="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"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN, revision="fp16", torch_dtype=torch.float16)
pipe.to(device)


print("hello")