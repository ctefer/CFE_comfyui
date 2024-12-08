import comfy
from .cfe_utils import build_sigmas

class CFE_Sigma_Scheduler:
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
    CATEGORY = "CFE/sampling"


    def execute(self, model, sampler_select, scheduler, steps, denoise):
        sigmas = build_sigmas(model, steps, scheduler, denoise)
        return (comfy.samplers.sampler_object(sampler_select), sigmas, )
    

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
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "output"
    CATEGORY = "CFE/sampling"



    def output(self, model, scheduler, steps, denoise):
        sigmas = build_sigmas(model, steps, scheduler, denoise)
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


    RETURN_TYPES = ("SAMPLER", "STRING", )
    RETURN_NAMES = ("sampler", "scheduler", )
    FUNCTION = "output"
    CATEGORY = "CFE/sampling"



    def output(self, sampler, scheduler):
        return (comfy.samplers.sampler_object(sampler), scheduler, )