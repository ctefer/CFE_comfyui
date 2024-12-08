import comfy
import torch


def find_lowest_64(value):
    """
        We use shifting to multiply by 64 and then divide by 64
        not sure I understand this, but there's an additional requirement to be around 24 points
        of the corrected value, so we add 23 to ensure any time we're >24 points we end up in the
        corrected pixel space for most resolutions
    """
    return (int(value + 23) >> 6) << 6

def build_sigmas(model, steps, scheduler, denoise):
    """
        Builds the sigmas from the steps and the denoise. 
    """
    sigmas = None
    if denoise <= 0.0:
        sigmas = torch.FloatTensor([])

    if sigmas is None:
        total_steps = int(steps/denoise)
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
        sigmas = sigmas[-(steps + 1):]

    return sigmas