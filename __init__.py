# from .mine_nodes import *
from .cfe_nodes import *


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "CFE_Aspect_Ratio" : CFE_Aspect_Ratio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CFE_Aspect_Ratio":"CFE Aspect Ratio"
}
