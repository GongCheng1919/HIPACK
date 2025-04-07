import os
import torch
from torch import nn
import time
import direct_conv2d 
from torch.nn.modules.conv import Conv2d
from torch.quantization import QConfig
from collections import defaultdict
from replace import replace_conv2d, generate_report
# disable FutureWarning
import warnings
import sys
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    diffusion_path = os.path.join(os.path.dirname(__file__), "diffusion")
    sys.path.insert(0, diffusion_path)
    print(f"Added to sys.path: {diffusion_path}")
    from denoising_diffusion_pytorch import Unet, GaussianDiffusion
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current sys.path: {sys.path}")

def run(mode: str):
    assert mode in ["float", "hipack", "qnnpack"], "mode must be one of 'float', 'hipack', or 'qnnpack'"
    
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = False
    )

    # Choose mode: 'float', 'hipack', or 'qnnpack'
    replace_conv2d(model, mode=mode, W_bits=3, A_bits=3)

    diffusion = GaussianDiffusion(
        model,
        image_size = 56,
        timesteps = 1    # number of steps
    )

    sampled_images = diffusion.sample(batch_size = 32, return_all_timesteps = False)
    print(sampled_images.shape)

import pandas as pd
import numpy as np

if __name__ == "__main__":
    run("hipack")
    run("qnnpack")

    generate_report("diffusion")
