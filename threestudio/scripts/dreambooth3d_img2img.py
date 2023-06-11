import requests
import torch
from PIL import Image
from io import BytesIO
import os
import numpy as np
import argparse

from diffusers import StableDiffusionImg2ImgPipeline, UNet2DConditionModel, DDIMScheduler

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    # add argument for image input folder
    parser.add_argument(
        "--input_folder",
        type=str,
        default=None,
        required=True,
        help="Path to folder containing input images.",
    )

    # add argument for prompt
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        required=True,
        help="Prompt for image generation.",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        required=True,
        help="Strength"
    )

    if input_args is not None:
            args = parser.parse_args(input_args)
    else:
            args = parser.parse_args()

    return args

# take in arguments from command line
args = parse_args()

device = "cuda"
model_id_or_path = "CompVis/stable-diffusion-v1-4"
unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch.float16
    )

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16, unet=unet, safety_checker=None)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

# response = requests.get(url)
# init_image = Image.open(BytesIO(response.content)).convert("RGB")

# iterate through folder and load png images
for filename in os.listdir(args.input_folder):
    if filename.endswith("_rgb.png"):
        print(filename)
        init_image = Image.open(args.input_folder + filename).convert("RGB")
        init_image = init_image.resize((512, 512))

        # init_image.save(args.input_folder + '/bu' + filename)

        prompt = args.prompt #"A fantasy landscape, trending on artstation"
        images = pipe(prompt=prompt, image=init_image, strength=args.strength, guidance_scale=7.5).images

        filename = filename.split('_')[0] + '.png'
        images[0].save(args.input_folder + '/' + filename)

# init_image = init_image.resize((768, 512))

# prompt = "A fantasy landscape, trending on artstation"

# images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
# images[0].save("fantasy_landscape.png")