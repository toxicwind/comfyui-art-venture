import os
import sys
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from segment_anything import SamPredictor


import folder_paths
import comfy.model_management
from comfy.utils import load_torch_file

print(os.path.join(os.path.dirname(__file__), "lama"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lama"))

from ..utils import tensor2pil, pil2tensor

models_path = folder_paths.models_dir
folder_paths.folder_names_and_paths["lama"] = (
    [os.path.join(models_path, "lama")],
    folder_paths.supported_pt_extensions,
)

LAMA_CONFIG = "https://huggingface.co/camenduru/big-lama/raw/main/big-lama/config.yaml"


class GetSAMEmbedding:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam_model": ("SAM_MODEL",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("SAM_EMBEDDING",)
    CATEGORY = "Art Venture/Inpainting"
    FUNCTION = "get_sam_embedding"

    def get_sam_embedding(self, image, sam_model):
        if sam_model.is_auto_mode:
            device = comfy.model_management.get_torch_device()
            sam_model.to(device=device)

        try:
            predictor = SamPredictor(sam_model)
            image = tensor2pil(image)
            image = image.convert("RGB")
            image = np.array(image)
            predictor.set_image(image, "RGB")
            embedding = predictor.get_image_embedding().cpu().numpy()

            print("embedding", embedding.shape)

        finally:
            if sam_model.is_auto_mode:
                sam_model.to(device="cpu")

        return (embedding,)


class SAMEmbeddingToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding": ("SAM_EMBEDDING",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Inpainting"
    FUNCTION = "sam_embedding_to_noise_image"

    def sam_embedding_to_noise_image(self, embedding: np.ndarray):
        # Flatten the array to a 1D array
        flat_arr = embedding.flatten()
        # Convert the 1D array to bytes
        bytes_arr = flat_arr.astype(np.float32).tobytes()
        # Convert bytes to RGBA PIL Image
        size = (embedding.shape[1], embedding.shape[2] * embedding.shape[3])

        img = Image.frombytes("RGBA", size, bytes_arr)

        return (pil2tensor(img),)


class LamaLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("lama"),),
                "device_mode": (["AUTO", "Prefer GPU", "CPU"],),
            },
        }

    RETURN_TYPES = ("LAMA_MODEL",)
    CATEGORY = "Art Venture/Loaders"
    FUNCTION = "load_lama"

    def load_lama(self, model_name, device_mode):
        import yaml
        from .lama.saicinpainting.training.trainers import make_training_model

        # Unless user explicitly wants to use CPU, we use GPU
        device = (
            comfy.model_management.get_torch_device()
            if device_mode == "Prefer GPU"
            else "CPU"
        )
        model_path = folder_paths.get_full_path("lama", model_name)
        train_config_path = os.path.join(os.path.dirname(model_path), "config.yaml")

        if not os.path.exists(train_config_path):
            # Download the config file
            import requests

            res = requests.get(LAMA_CONFIG, allow_redirects=True)
            with open(train_config_path, "wb") as f:
                f.write(res.content)

        with open(train_config_path, "r") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"

        model: torch.nn.Module = make_training_model(train_config)
        state_dict = load_torch_file(model_path)
        model.load_state_dict(state_dict, strict=False)

        if device_mode == "Prefer GPU":
            model.to(device)

        model.freeze()
        model.is_auto_mode = device_mode == "AUTO"
        model.is_cpu_mode = device_mode == "CPU"

        return (model,)


class LamaInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lama_model": ("LAMA_MODEL",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_dialation": (
                    "INT",
                    {"min": 0, "max": 1000, "step": 1, "default": 0},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Inpainting"
    FUNCTION = "lama_inpaint"

    @torch.no_grad()
    def lama_inpaint(
        self,
        lama_model: torch.nn.Module,
        image: torch.Tensor,
        mask: torch.Tensor,
        mask_dialation=0,
    ):
        from .lama.saicinpainting.evaluation.data import pad_tensor_to_modulo
        from .lama.saicinpainting.evaluation.utils import move_to_device
        from .inpaint_anything.utils import dilate_mask

        try:
            mask = mask.cpu().numpy()
            if mask_dialation > 0:
                mask = dilate_mask(mask, mask_dialation)
            if np.max(mask) <= 1:
                print("scale mask to 255")
                mask = mask * 255

            img = image.float()
            mask = torch.from_numpy(mask).float()
            print("img", img.shape, img)

            batch = {}
            batch["image"] = img.permute(0, 3, 1, 2)
            batch["mask"] = mask[None, None]
            unpad_to_size = [batch["image"].shape[2], batch["image"].shape[3]]
            batch["image"] = pad_tensor_to_modulo(batch["image"], 8)
            batch["mask"] = pad_tensor_to_modulo(batch["mask"], 8)

            device = comfy.model_management.get_torch_device()
            if lama_model.is_auto_mode:
                lama_model.to(device)
            if not lama_model.is_cpu_mode:
                batch = move_to_device(batch, device)

            batch["mask"] = (batch["mask"] > 0) * 1
            batch = lama_model(batch)
            res = batch["inpainted"].permute(0, 2, 3, 1)
            res = res.detach().cpu().numpy()

            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                res = res[:, :orig_height, :orig_width, :]

            return (torch.from_numpy(res),)
        finally:
            if lama_model.is_auto_mode:
                lama_model.to(device="cpu")
