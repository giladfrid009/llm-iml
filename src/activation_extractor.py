import torch
from torch import nn, Tensor
from contextlib import contextmanager
import warnings
from typing import Callable


class ActivationExtractor:
    """
    Extracts activations from specified layers of a PyTorch model.

    Usage:
        # Basic usage
        extractor = ActivationExtractor(model, 'layer1.0.conv1', 'layer2.0.conv1')
        with extractor.capture():
            output = model(input_tensor)
        activations = extractor.get_activations()
    """

    def __init__(
        self,
        model: nn.Module,
        *layer_specs: str | type,
        exact_match: bool = True,
        capture_output: bool = True,
    ):
        """
        Initialize the activation extractor.

        Args:
            model: PyTorch model to extract activations from
            layer_specs: Layer names or types to capture activations from
            exact_match: If True, requires exact layer name matches
            capture_output: If True, captures the output of the layer. If False, captures the input.
        """
        if len(layer_specs) == 0:
            raise ValueError("At least one layer must be specified.")

        self.model = model
        self.layer_specs = layer_specs
        self.exact_match = exact_match
        self.capture_output = capture_output

        self._activations: dict[str, Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._layers = self._get_layers()

    @property
    def layer_names(self) -> list[str]:
        """The full names of the layers being monitored, ordered by their appearance in the model."""
        return [name for name, layer in self._layers]

    def get_hparams(self) -> dict:
        """Returns hyperparameters for the activation extractor."""
        return {
            "activ_extractor/exact_match": self.exact_match,
            "activ_extractor/capture_output": self.capture_output,
            "activ_extractor/layer_specs": str(self.layer_specs),
            "activ_extractor/layer_names": str(self.layer_names),
        }

    def get_activations(self) -> dict[str, Tensor]:
        """Returns the current captured layer activations."""
        if len(self._activations) == 0:
            warnings.warn("No activations captured yet. Did you forget to call `capture()`?")
        return {k: v.clone() for k, v in self._activations.items()}

    @contextmanager
    def capture(self):
        """
        Context manager to capture activations from the specified layers.
        """
        try:
            self._activations.clear()  # Clear any previous activations
            self._register_hooks()
            yield self
        finally:
            self._remove_hooks()

    def _find_layers_by_type(self, types: list[type]) -> list[str]:
        """Find all layers in the model matching the specified types, and return their full names."""
        return [name for name, module in self.model.named_modules() if isinstance(module, tuple(types))]

    def _find_layers_by_name(self, names: list[str], exact_match: bool = True) -> list[str]:
        """Find layers matching the specified names, and return their full names."""
        layers = []
        for name, _ in self.model.named_modules():
            search_name = name if exact_match else name.split(".")[-1]
            if search_name in names:
                layers.append(name)

        missing = set(names) - {(n if exact_match else n.split(".")[-1]) for n in layers}
        if missing and exact_match:
            warnings.warn(f"Layers not found: {missing}")

        return layers

    def _get_layers(self) -> list[tuple[str, nn.Module]]:
        """Returns a list of (layer_name, layer_module) tuples for the specified layers."""
        name_specs = [l for l in self.layer_specs if isinstance(l, str)]
        type_specs = [l for l in self.layer_specs if isinstance(l, type) and issubclass(l, nn.Module)]

        layers_names = set()
        if len(name_specs) > 0:
            layers_names.update(self._find_layers_by_name(name_specs, self.exact_match))
        if len(type_specs) > 0:
            layers_names.update(self._find_layers_by_type(type_specs))

        layers = []
        for name, module in self.model.named_modules():
            if name in layers_names:
                layers.append((name, module))

        if len(layers) == 0:
            raise ValueError("No layers found to monitor")

        return layers

    def _create_output_hook(self, layer_name: str):
        """Create a forward hook to capture layer output."""

        def hook_fn(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor):
            self._activations[layer_name] = output

        return hook_fn

    def _create_input_hook(self, layer_name: str):
        """Create a forward hook to capture layer input."""

        def hook_fn(module: nn.Module, inputs: tuple[Tensor, ...]):
            tensors = [item for item in inputs if isinstance(item, torch.Tensor)]
            if not len(tensors) == 1:
                raise ValueError(f"Expected 1 input tensor, got {len(tensors)}")
            self._activations[layer_name] = tensors[0]

        return hook_fn

    def _register_hooks(self) -> None:
        """Register forward hooks for the specified layers."""
        for layer_name, layer_module in self._layers:

            if self.capture_output:
                handle = layer_module.register_forward_hook(self._create_output_hook(layer_name))
            else:
                handle = layer_module.register_forward_pre_hook(self._create_input_hook(layer_name))

            self._handles.append(handle)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __del__(self):
        """Clean up hooks when object is deleted."""
        self._remove_hooks()


class ActivationLoss(torch.nn.Module):
    def __init__(
        self,
        loss_fn: Callable[..., torch.Tensor],
        aggr_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        """
        A loss module for aggregating per-layer activation losses into one final scalar.

        This class is useful when a model produces intermediate activations from multiple
        layers and you want to compute a loss term at each layer, then combine (aggregate)
        those layerwise losses into a single scalar loss.

        Args:
            loss_fn (Callable[..., torch.Tensor]):
                A function that computes the loss for a given layer across input activation
                dictionaries. It is called once per layer.
                - It must accept one or more Tensors (corresponding to the same layer key
                  across multiple input dictionaries) and return a Tensor of shape
                  ``(batch_size,)``, representing the per-sample loss for that layer.

            aggr_fn (Callable[[torch.Tensor], torch.Tensor], optional):
                A function that aggregates the losses across all layers for each sample.
                - Its input is a Tensor of shape ``(batch_size, num_layers)``. Element
                  ``losses[i, j]`` corresponds to the loss of the i-th sample at the j-th
                  layer.
                - Its output must be a Tensor of shape ``(batch_size,)``, representing
                  the aggregated per-sample loss across all layers.
                - Defaults to summing losses of all layers for each sample.
        """
        super().__init__()

        if aggr_fn is None:
            aggr_fn = lambda losses: torch.sum(losses, dim=-1)

        self.loss_fn = loss_fn
        self.aggr_fn = aggr_fn

    def forward(self, *args: dict[str, torch.Tensor]) -> torch.Tensor:
        keys = args[0].keys()
        sample = next(iter(args[0].values()))
        losses = torch.empty(size=(sample.size(0), len(keys)), device=sample.device)
        for i, key in enumerate(keys):
            losses[:, i] = self.loss_fn(*[arg[key].flatten(1) for arg in args])
        return self.aggr_fn(losses).mean()
