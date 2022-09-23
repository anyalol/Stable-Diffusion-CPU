import gradio as gr
from PIL import Image
import os
MY_SECRET_TOKEN="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"

from diffusers import StableDiffusionPipeline
YOUR_TOKEN=MY_SECRET_TOKEN
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN)
pipe.to(device)
