import comfy
import comfy.model_management
import comfy.samplers
import node_helpers

from .cfe_utils import *


# NOTES:
## 
# Artifacts
## 
# Scheduler: this is the named scheduler, a string value
# Sigmas: this is the value being fed to the sampler which comprises of the scheduler (name), 
#   steps, denoise, and model

def _flux_sampler(model, noise, cfg, cond, sampler, sigmas, latent_image):
    """
        Simple sampler function for reuse
    """
    latent = latent_image
    latent_image = latent["samples"]
    latent = latent.copy()
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
    latent["samples"] = latent_image

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]


    # update the conditioner value
    condition = node_helpers.conditioning_set_values(cond, {"guidance": cfg})
    guider = comfy.samplers.CFGGuider(model)
    guider.inner_set_conds({"positive": condition})


    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, 
                            denoise_mask=noise_mask, disable_pbar=disable_pbar, seed=noise.seed)
    
    samples = samples.to(comfy.model_management.intermediate_device())
    latent["samples"] = samples

    return latent



class CFE_Flux_In_Pipe:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "clip": ("CLIP", {"tooltip": "The clip used for text encoding."}),
                "vae": ("VAE", {"tooltip": "The latent image to denoise."}),
                "cond": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "sampler": ("SAMPLER", {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": ("STRING", {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
            },
        }
    
    RETURN_TYPES = ("FLUX_PIPE",)
    RETURN_NAMES = ("pipe",)

    FUNCTION = "in_pipe"
    CATEGORY = "CFE/flux"

    def in_pipe(self, model, clip, vae, cond, sampler, scheduler):
        pipe = (model, clip, vae, cond, sampler, scheduler)
        return (pipe, )
    

class CFE_Flux_Out_Pipe:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("FLUX_PIPE", {"tooltip": "The bus line carrying the data."}),
            },
        }
    
    RETURN_TYPES = ("FLUX_PIPE", "MODEL", "CLIP", "VAE", "CONDITIONING", "SAMPLER", "STRING")
    RETURN_NAMES = ("pipe", "model", "clip", "vae", "cond", "sampler", "scheduler" )

    FUNCTION = "out_pipe"
    CATEGORY = "CFE/flux"

    def out_pipe(self, pipe):
        model, clip, vae, cond, sampler, scheduler = pipe
        return (pipe, model, clip, vae, cond, sampler, scheduler)
    



class CFE_Flux_Guidance:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                "clip" : ("CLIP", {"tooltip" : "The name of the sampler being used"}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
            }
        }


    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("cond",)
    FUNCTION = "output"
    CATEGORY = "CFE/flux"

    def output(self, clip, guidance, text):
        
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        output["guidance"] = guidance
        
        return ([[cond, output]], )


class CFE_FLUX_Pipe_Sampler:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("FLUX_PIPE", {"tooltip": "The bus line carrying the data."}),
                "noise":("NOISE", {"tooltip": "Noise for generation"},),
                "latent_image": ("LATENT", {"tooltip": "The latent image."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
            },
        }

    RETURN_TYPES = ("FLUX_PIPE", "LATENT","VAE")
    RETURN_NAMES = ("pipe", "latent", "vae")
    FUNCTION = "execute"
    CATEGORY = "CFE/flux"

    def execute(self, pipe, noise, latent_image, cfg, steps, denoise):
        model, _, vae, cond, sampler, scheduler = pipe
        sigmas = build_sigmas(model, steps, scheduler, denoise)
        out = _flux_sampler(model, noise, cfg, cond, sampler, sigmas, latent_image)
        return (pipe, out, vae)


class CFE_FLUX_Sampler:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise":("NOISE", {"tooltip": "Noise for generation"},),
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "vae": ("VAE", {"tooltip": "The latent image to denoise."}),
                "cond": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "sampler": ("SAMPLER", {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "sigmas": ("SIGMAS", {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
            },
        }

    RETURN_TYPES = ("LATENT", "VAE",)
    RETURN_NAMES = ("latent", "vae", )
    FUNCTION = "execute"
    CATEGORY = "CFE/flux"

    def execute(self, noise, model, vae, cond, sampler, sigmas, latent_image, cfg):
        out = _flux_sampler(model, noise, cfg, cond, sampler, sigmas, latent_image)
        return (out, vae, )
