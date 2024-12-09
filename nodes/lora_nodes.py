import os
import folder_paths

# Must of this was stolen from rgthree Power Lora Loader, because it was awesome


# pylint: disable = too-many-return-statements, too-many-branches
def get_lora_by_filename(file_path, lora_paths=None, log_node=None):
  """Returns a lora by filename, looking for exactl paths and then fuzzier matching. """
  lora_paths = lora_paths if lora_paths is not None else folder_paths.get_filename_list('loras')

  if file_path in lora_paths:
    return file_path

  lora_paths_no_ext = [os.path.splitext(x)[0] for x in lora_paths]

  # See if we've entered the exact path, but without the extension
  if file_path in lora_paths_no_ext:
    found = lora_paths[lora_paths_no_ext.index(file_path)]
    return found

  # Same check, but ensure file_path is without extension.
  file_path_force_no_ext = os.path.splitext(file_path)[0]
  if file_path_force_no_ext in lora_paths_no_ext:
    found = lora_paths[lora_paths_no_ext.index(file_path_force_no_ext)]
    return found

  # See if we passed just the name, without paths.
  lora_filenames_only = [os.path.basename(x) for x in lora_paths]
  if file_path in lora_filenames_only:
    found = lora_paths[lora_filenames_only.index(file_path)]
    return found

  # Same, but force the input to be without paths
  file_path_force_filename = os.path.basename(file_path)
  lora_filenames_only = [os.path.basename(x) for x in lora_paths]
  if file_path_force_filename in lora_filenames_only:
    found = lora_paths[lora_filenames_only.index(file_path_force_filename)]
    return found

  # Check the filenames and without extension.
  lora_filenames_and_no_ext = [os.path.splitext(os.path.basename(x))[0] for x in lora_paths]
  if file_path in lora_filenames_and_no_ext:
    found = lora_paths[lora_filenames_and_no_ext.index(file_path)]
    return found

  # And, one last forcing the input to be the same
  file_path_force_filename_and_no_ext = os.path.splitext(os.path.basename(file_path))[0]
  if file_path_force_filename_and_no_ext in lora_filenames_and_no_ext:
    found = lora_paths[lora_filenames_and_no_ext.index(file_path_force_filename_and_no_ext)]
    return found

  # Finally, super fuzzy, we'll just check if the input exists in the path at all.
  for index, lora_path in enumerate(lora_paths):
    if file_path in lora_path:
      found = lora_paths[index]
      return found


  return None

class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

  def __ne__(self, __value: object) -> bool:
    return False

class FlexibleOptionalInputType(dict):
  """A special class to make flexible nodes that pass data to our python handlers.

  Enables both flexible/dynamic input types (like for Any Switch) or a dynamic number of inputs
  (like for Any Switch, Context Switch, Context Merge, Power Lora Loader, etc).

  Note, for ComfyUI, all that's needed is the `__contains__` override below, which tells ComfyUI
  that our node will handle the input, regardless of what it is.

  However, with https://github.com/comfyanonymous/ComfyUI/pull/2666 a large change would occur
  requiring more details on the input itself. There, we need to return a list/tuple where the first
  item is the type. This can be a real type, or use the AnyType for additional flexibility.

  This should be forwards compatible unless more changes occur in the PR.
  """
  def __init__(self, type):
    self.type = type

  def __getitem__(self, key):
    return (self.type, )

  def __contains__(self, key):
    return True
  
any_type = AnyType("*")
  
class CFE_Lora_Params:
  """ The Power Lora Loader is a powerful, flexible node to add multiple loras for distribution """

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      # Since we will pass any number of loras in from the UI, this needs to always allow an
      "optional": FlexibleOptionalInputType(any_type),
      "hidden": {},
    }

  RETURN_TYPES = ("LORA_PARAMS",)
  RETURN_NAMES = ("loras",)
  FUNCTION = "get_loras"

  CATEGORY = "CFE/loras"

  def get_loras(self, **kwargs):
    """Loops over the provided loras in kwargs and applies valid ones."""
    loras = {"loras":[], "strengths":[]}
    for key, value in kwargs.items():
      key = key.upper()
      if key.startswith('LORA_') and 'on' in value and 'lora' in value and 'strength' in value:
        
        strength_model = value['strength']
        # If we just passed one strength value, then use it for both, if we passed a strengthTwo
        # as well, then our `strength` will be for the model, and `strengthTwo` for clip.
        strength_clip = value['strengthTwo'] if 'strengthTwo' in value and value[
          'strengthTwo'] is not None else strength_model
        
        loras[""]
        if value['on'] and (strength_model != 0 or strength_clip != 0):
          lora = get_lora_by_filename(value['lora'], log_node=self.NAME)

          loras["loras"].append(lora)
          loras["strengths"].append([strength_model, strength_clip])

    return (loras,)