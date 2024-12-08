import math
import torch
import comfy
import comfy.model_management
import comfy.samplers
import node_helpers


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

def _find_lowest_64(value):
    """
        We use shifting to multiply by 64 and then divide by 64
        not sure I understand this, but there's an additional requirement to be around 24 points
        of the corrected value, so we add 23 to ensure any time we're >24 points we end up in the
        corrected pixel space for most resolutions
    """
    return (int(value + 23) >> 6) << 6

def _build_sigmas(model, steps, scheduler, denoise):
    """
        Builds the sigmas from the steps and the denoise. 
    """
    sigmas = None
    if denoise < 1.0:
        if denoise <= 0.0:
            sigmas = torch.FloatTensor([])
        total_steps = int(steps/denoise)

    if sigmas is None:
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
        sigmas = sigmas[-(steps + 1):]

    return sigmas


class CFE_Aspect_Ratio:
    """
        Creates an empty latent image that will more easily characterize the resolution process for SD1.5, SDXL, and FLUX
    """
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "type" : (["square", "portrait (w:h)", "landscape (h:w)"],),
                "resolution": ([
                    "1:1",
                    "2:3", 
                    "3:4", 
                    "4:7",
                    "7:9",
                    "8:15",
                    "9:16",
                    "9:21",
                ],),
                "megapixel": (["1 MP (SDXL)", "2 MP (FLUX)", ".5 MP (SD1.5)"],),
                "clip_size": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Multiplies the output width and height values",
                    "lazy" : True
                }),
                "batch_size": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 4096, 
                    "tooltip": "The number of latent images in the batch.",
                    "lazy" : True
                }),
            },
        }

    RETURN_TYPES = ("LATENT","INT","INT")
    RETURN_NAMES = ("latent", "clip_width","clip_height")

    FUNCTION = "calculate_resolution"

    CATEGORY = "CFE"

   

    def calculate_resolution(self, type, resolution, megapixel, clip_size, batch_size):

        if type == "square":
            ratio = 1
        else:
            colon = resolution.find(":")
            res0 = float(resolution[:colon])
            res1 = float(resolution[colon + 1:])
            ratio = res1/ res0
        
        mp_mark = megapixel[:1]

        if mp_mark == "1":
            mp = 1024 * 1024
        elif mp_mark == "2":
            mp = 2 * 1024 * 1024
        else:
            mp = 512 * 512

        large = int(math.sqrt(math.floor(ratio * mp)))
        large = _find_lowest_64(large)
        
        small =  int(large / ratio)
        small =  _find_lowest_64(small)
        
        if type.find("portrait") == -1:
            clip_width = large
            clip_height = small
        else:
            clip_width = small
            clip_height = large

        latent = torch.zeros([batch_size, 4, clip_height // 8, clip_width // 8], device=self.device)

        clip_width *= clip_size
        clip_width = _find_lowest_64(clip_width)
        clip_height *= clip_size
        clip_height = _find_lowest_64(clip_height)



        return ({"samples":latent}, clip_width, clip_height)
    

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
    CATEGORY = "CFE"

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
    CATEGORY = "CFE"

    def out_pipe(self, pipe):
        model, clip, vae, cond, sampler, scheduler = pipe
        return (pipe, model, clip, vae, cond, sampler, scheduler)
    

class CFE_Scheduler:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                "model" : ("MODEL", {"tooltip" : "The model used for denoising the input latent"}),
                "scheduler": ("STRING", {"default": "euler", "tooltip" : "The name of the scheduler being used"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }


    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "output"
    CATEGORY = "CFE"



    def output(self, model, scheduler, steps, denoise):
        sigmas = _build_sigmas(model, steps, scheduler, denoise)
        return (sigmas, )

class CFE_Sigma_Sampler_Strings:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                "sampler" : (comfy.samplers.SAMPLER_NAMES, {"tooltip" : "The name of the sampler being used"}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"tooltip" : "The name of the scheduler being used"}),
            }
        }


    RETURN_TYPES = ("SAMPLER", "STRING")
    RETURN_NAMES = ("sampler", "scheduler")
    FUNCTION = "output"
    CATEGORY = "CFE"



    def output(self, sampler_select, scheduler):
        return (comfy.samplers.sampler_object(sampler_select), scheduler, )

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
    CATEGORY = "CFE"

    def output(self, clip, guidance, text):
        
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        node_helpers.conditioning_set_values(cond, {"guidance": guidance})
        
        return ([[cond, output]], )

class CFE_Sigma_Sampler:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                "model" : ("MODEL", {"tooltip" : "The model used for denoising the input latent"}),
                "sampler_select" : (comfy.samplers.SAMPLER_NAMES, {"tooltip" : "The name of the sampler being used"}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"tooltip" : "The name of the scheduler being used"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }


    RETURN_TYPES = ("SAMPLER", "SIGMAS")
    RETURN_NAMES = ("sampler", "sigmas")
    FUNCTION = "execute"
    CATEGORY = "CFE"


    def execute(self, model, sampler_select, scheduler, steps, denoise):
        sigmas = _build_sigmas(model, steps, scheduler, denoise)
        return (comfy.samplers.sampler_object(sampler_select), sigmas, )





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
    CATEGORY = "CFE"

    def execute(self, pipe, noise, latent_image, cfg, steps, denoise):
        model, _, vae, cond, sampler, scheduler = pipe
        sigmas = _build_sigmas(model, steps, scheduler, denoise)
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
    CATEGORY = "CFE"

    def execute(self, noise, model, vae, cond, sampler, sigmas, latent_image, cfg):
        out = _flux_sampler(model, noise, cfg, cond, sampler, sigmas, latent_image)
        return (out, vae, )
