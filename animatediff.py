import os
import torch
import numpy as np
from PIL import Image

import folder_paths
import comfy.ldm.modules.diffusionmodules.openaimodel as openaimodel
import comfy.model_management as model_management
from comfy.ldm.modules.attention import SpatialTransformer
from comfy.utils import load_torch_file
from comfy.sd import ModelPatcher, calculate_parameters

from .modules.log import logger
from .modules.motion_module import MotionWrapper, VanillaTemporalModule


orig_forward_timestep_embed = openaimodel.forward_timestep_embed


def forward_timestep_embed(
    ts, x, emb, context=None, transformer_options={}, output_shape=None
):
    for layer in ts:
        if isinstance(layer, openaimodel.TimestepBlock):
            x = layer(x, emb)
        elif isinstance(layer, VanillaTemporalModule):
            x = layer(x, context)
        elif isinstance(layer, (SpatialTransformer, VanillaTemporalModule)):
            x = layer(x, context, transformer_options)
            transformer_options["current_index"] += 1
        elif isinstance(layer, openaimodel.Upsample):
            x = layer(x, output_shape=output_shape)
        else:
            x = layer(x)
    return x


openaimodel.forward_timestep_embed = forward_timestep_embed

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(CURRENT_DIR, "models")
allowed_model_files = {"mm_sd_v14.ckpt", "mm_sd_v15.ckpt"}
motion_module: MotionWrapper = None


def get_model_files():
    model_files = [
        m for m in allowed_model_files if os.path.isfile(os.path.join(model_dir, m))
    ]

    return model_files


class AnimateDiffLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "model_name": (get_model_files(),),
                "width": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 8}),
                "frame_number": (
                    "INT",
                    {"default": 16, "min": 2, "max": 24, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "LATENT")
    CATEGORY = "Animate Diff"
    FUNCTION = "inject_motion_modules"

    def inject_motion_modules(
        self,
        model: ModelPatcher,
        model_name: str,
        width: int,
        height: int,
        frame_number=16,
    ):
        model = model.clone()
        model_path = os.path.join(model_dir, model_name)

        global motion_module
        if motion_module is None:
            logger.info(f"Loading motion module {model_name} from {model_path}")
            mm_state_dict = load_torch_file(model_path)
            motion_module = MotionWrapper()

            parameters = calculate_parameters(mm_state_dict, "")
            usefp16 = model_management.should_use_fp16(model_params=parameters)
            if usefp16:
                print("Using fp16, converting motion module to fp16")
                motion_module.half()
            offload_device = model_management.unet_offload_device()
            motion_module = motion_module.to(offload_device)
            motion_module.load_state_dict(mm_state_dict)

        unet = model.model.diffusion_model

        logger.info(f"Injecting motion module into UNet input blocks.")
        for mm_idx, unet_idx in enumerate([1, 2, 4, 5, 7, 8, 10, 11]):
            mm_idx0, mm_idx1 = mm_idx // 2, mm_idx % 2
            unet.input_blocks[unet_idx].append(
                motion_module.down_blocks[mm_idx0].motion_modules[mm_idx1]
            )

        logger.info(f"Injecting motion module into UNet output blocks.")
        for unet_idx in range(12):
            mm_idx0, mm_idx1 = unet_idx // 3, unet_idx % 3
            if unet_idx % 2 == 2:
                unet.output_blocks[unet_idx].insert(
                    -1, motion_module.up_blocks[mm_idx0].motion_modules[mm_idx]
                )
            else:
                unet.output_blocks[unet_idx].append(
                    motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1]
                )

        latent = torch.zeros([frame_number, 4, width // 8, height // 8]).cpu()

        return (model, {"samples": latent})


class AnimateDiffCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "INT",
                    {"default": 8, "min": 1, "max": 24, "step": 1},
                ),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "save_image": (("Enabled", "Disabled"),),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Animate Diff"
    OUTPUT_NODE = True
    FUNCTION = "generate_gif"

    def generate_gif(
        self,
        images,
        frame_rate: int,
        loop_count: int,
        save_image="Enabled",
        filename_prefix="AnimateDiff",
    ):
        import imageio

        # convert images to numpy
        image_nps = []
        for image in images:
            img = 255.0 * image.cpu().numpy()
            img = np.clip(img, 0, 255).astype(np.uint8)
            image_nps.append(img)

        # save image
        output_dir = (
            folder_paths.get_output_directory()
            if save_image == "Enabled"
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)
        file = f"{filename}_{counter:05}.gif"
        file_path = os.path.join(full_output_folder, file)

        # save gif
        imageio.mimsave(
            file_path,
            image_nps,
            duration=round(1000 / frame_rate),
            loop=loop_count,
        )

        # load saved image back as torch tensor
        saved = Image.open(file_path)
        saved = saved.convert("RGB")
        saved = np.array(saved).astype(np.float32) / 255.0
        saved = torch.from_numpy(saved)[None,]

        previews = [
            {
                "filename": file,
                "subfolder": subfolder,
                "type": "output" if save_image == "Enabled" else "temp",
            }
        ]

        return {"ui": {"images": previews}, "result": (saved,)}


NODE_CLASS_MAPPINGS = {
    "AnimateDiffLoader": AnimateDiffLoader,
    "AnimateDiffCombile": AnimateDiffCombine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimateDiffLoader": "Animate Diff Loader",
    "AnimateDiffCombile": "Animate Diff Combine",
}
