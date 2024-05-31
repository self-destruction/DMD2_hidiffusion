from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler, AutoencoderTiny
from accelerate import Accelerator
import random
import numpy as np
import argparse
import torch
import time
from PIL import Image
from hidiffusion import apply_hidiffusion
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

class ModelWrapper:
    def __init__(self, args, accelerator):
        super().__init__()
        # disable all gradient calculations
        torch.set_grad_enabled(False)
        
        if args.precision == "bfloat16":
            DTYPE = torch.bfloat16
        elif args.precision == "float16":
            DTYPE = torch.float16
        else:
            DTYPE = torch.float32
        device = accelerator.device

        unet = UNet2DConditionModel.from_pretrained('RunDiffusion/Juggernaut-X-v10', subfolder='unet').to(device, DTYPE)
        unet.load_state_dict(torch.load(hf_hub_download(args.repo_name, args.ckpt_name), map_location=device))

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            args.model_id, unet=unet, torch_dtype=DTYPE, variant='fp16', 
            use_safetensors=True,
            add_watermarker=False,
        ).to(device)
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        if args.fast_vae_decode:
            self.pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_dtype=DTYPE).to(device)

        apply_hidiffusion(self.pipe)
        # pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_tiling()


    @staticmethod
    def _get_time():
        torch.cuda.synchronize()
        return time.time()


    @torch.no_grad()
    def inference(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_images: int,
        num_step: int
    ):
        print("Running model inference...")

        if seed == -1:
            seed = np.random.randint(0, np.iinfo(np.int32).max)

        generator = torch.manual_seed(seed)

        start_time = self._get_time()

        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_step,
            guidance_scale=0.0,
            eta=1.0,
            num_images_per_prompt=num_images,
            height=height,
            width=width,
            generator=torch.manual_seed(seed),
        ).images

        end_time = self._get_time()

        print(f"run successfully in {(end_time-start_time):.2f} seconds")

        return images


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="RunDiffusion/Juggernaut-XL-v9")
parser.add_argument("--precision", type=str, default="float16", choices=["float32", "float16", "bfloat16"])
parser.add_argument("--num_step", type=int, default=4, choices=[1, 4])
parser.add_argument("--repo_name", type=str, default="tianweiy/DMD2")
parser.add_argument("--ckpt_name", type=str, default="dmd2_sdxl_4step_unet_fp16.bin")
# Use Tiny VAE for faster decoding
parser.add_argument("--fast_vae_decode", action='store_true', default=False)

parser.add_argument("--save_dir", type=str)
parser.add_argument("--save_file_name", type=str)

parser.add_argument("--prompt", type=str, default='A photo of a cat')
parser.add_argument("--negative_prompt", type=str, default='blurry, ugly, duplicate, poorly drawn, deformed, mosaic')
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--num_images", type=int, default=1, choices=range(1, 16))
# Height
parser.add_argument("--h", type=int, default=1024)
# Width
parser.add_argument("--w", type=int, default=1024)

args = parser.parse_args()
print(args)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True 

accelerator = Accelerator()
model = ModelWrapper(args, accelerator)

ims = model.inference(
    args.prompt,
    args.negative_prompt,
    args.seed,
    args.h,
    args.w,
    args.num_images,
    args.num_step
)

for i, img in enumerate(ims):
    img.save(f'{args.save_dir}/{args.save_file_name}_{i}.png')
