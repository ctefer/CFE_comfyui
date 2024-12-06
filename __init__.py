# from .mine_nodes import *
from .cfe_nodes import *


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "CFE Aspect Ratio" : CFE_Aspect_Ratio,
    "CFE Flux In Pipe" : CFE_Flux_In_Pipe,
    "CFE Flux Out Pipe" : CFE_Flux_Out_Pipe,
    "CFE FLUX Sampler" : CFE_FLUX_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CFE_Aspect_Ratio":"CFE Aspect Ratio",
    "CFE_Flux_In_Pipe":"CFE Flux In Pipe",
    "CFE_Flux_Out_Pipe":"CFE Flux Out Pipe",
    "CFE_FLUX_Sampler":"CFE Flux Sampler",
}
