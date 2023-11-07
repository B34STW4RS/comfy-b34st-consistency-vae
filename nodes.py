from .consistencydecoder import ConsistencyDecoder
from PIL import Image
import torch
import numpy as np
import os

#fixed to support batches and for some reason the ram requirements went way down and can now use it for AD, whoopee.

#proper conversion
def np2tensor(img_np: np.ndarray | list[np.ndarray]) -> torch.Tensor:
    if isinstance(img_np, list):
        return torch.cat([np2tensor(img) for img in img_np], dim=0)

    return torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)


pwd = os.getcwd()

#where the magic happens, sorta
class itsConsistency:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent"


    def decode(self, latent):
        # Instantiate the decoder//why not?
        decoder_consistency = ConsistencyDecoder(device="cuda:0", download_root=pwd)

        # Ensure the latent is a batch by checking dimensions
        latents = latent["samples"].to("cuda:0")
        if len(latents.size()) == 1:
            latents = latents.unsqueeze(0)

        # we hold stuff here
        image_arrays = []

        # Process each latent in the biatch
        for latent_sample in latents:
            # Decode and process each latent sample
            consistent_latent = decoder_consistency(latent_sample.unsqueeze(0))
            image = consistent_latent.squeeze(0).cpu().numpy()
            
            # Ensure the image is in the range [0, 255] and correct shape [H, W, C]//is this right? dunno.
            image = np.clip((image + 1.0) * 127.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)
            image_arrays.append(image)

        # Convert the list of numpy arrays to a batched tensor
        image_tensor = np2tensor(image_arrays)

        # Return the image tensor as a tuple//always forget this
        return (image_tensor,)



NODE_CLASS_MAPPINGS = {
	"Comfy_itsConsistencyVAE": itsConsistency,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"Comfy_itsConsistencyVAE": "its Consistency VAE Decoder",
}
