from transformers.generation.configuration_utils import GenerationConfig
from typing import Any


class GenConfig:
    """
    A configuration class for generation parameters, similar to `GenerationConfig` from Hugging Face Transformers.
    
    Important: 
    When generating, for all parameters marked as `None` the model default generation config value will be used.
    """
    def __init__(
        self,
        *,
        max_length: int | None = None,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        max_time: float | None = None,
        **kwargs,
    ):
        self.max_length = max_length
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.max_time = max_time

        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_hparams(self) -> dict[str, Any]:
        """
        Returns a dictionary of all non-None parameters in this config.
        """
        params = {}
        for name, value in vars(self).items():
            if not name.startswith("_") and value is not None:
                params[name] = value
        return params

    def patch_params(
        self,
        generation_config: GenerationConfig | None = None,
        param_dict: dict[str, Any] | None = None,
    ):
        """
        Update any parameter in the provided arguments with non-None values from this config.

        Args:
            generation_config (GenerationConfig | None): The GenerationConfig object to update.
            param_dict (dict[str, Any] | None): A dictionary of parameters to update.
        """

        params = self.get_hparams()

        if generation_config is not None:
            for name, value in params.items():
                if hasattr(generation_config, name):
                    setattr(generation_config, name, value)
                else:
                    raise ValueError(f"Field '{name}' not found in GenerationConfig.")

        if param_dict is not None:
            for name, value in params:
                if name in param_dict:
                    param_dict[name] = value

    def update(self, **kwargs) -> dict[str, Any]:
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`): Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs
