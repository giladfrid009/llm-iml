import torch
from torch import nn, Tensor
from contextlib import contextmanager
from typing import Callable
from src.utils.logging import create_logger
from typing import Any

logger = create_logger(__name__)


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
            "exact_match": self.exact_match,
            "capture_output": self.capture_output,
            "layer_specs": str(self.layer_specs),
            "layer_names": str(self.layer_names),
        }

    def get_activations(self, clone: bool = False) -> dict[str, Tensor]:
        """Returns the current captured layer activations."""
        if len(self._activations) == 0:
            logger.warning("No activations captured yet. Did you forget to call `capture()`?")
            return {}

        if clone:
            return {k: v.clone() for k, v in self._activations.items()}
        return self._activations.copy()

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
            logger.warning(f"Layers not found: {missing}")

        return layers

    def _get_layers(self) -> list[tuple[str, nn.Module]]:
        """Returns a list of (layer_name, layer_module) tuples for the specified layers."""
        name_specs = [n for n in self.layer_specs if isinstance(n, str)]
        type_specs = [t for t in self.layer_specs if isinstance(t, type) and issubclass(t, nn.Module)]

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

        def hook_fn(module: nn.Module, args: tuple[Tensor, ...], output: tuple[Tensor, ...] | Tensor):
            if isinstance(output, tuple):
                logger.debug(f"[Layer {layer_name}]: Output is a tuple; using the first element.")
                output = output[0]

            if not isinstance(output, torch.Tensor):
                raise ValueError(f"[Layer {layer_name}]: Expected output to be a Tensor, got {type(output)}")

            self._activations[layer_name] = output

        return hook_fn

    def _create_input_hook(self, layer_name: str):
        """Create a forward hook to capture layer input."""

        def hook_fn(module: nn.Module, args: tuple[Tensor, ...], kwargs: dict[str, Any]):
            tensors_args = [item for item in args if isinstance(item, torch.Tensor)]
            tensors_kwargs = [item for item in kwargs.values() if isinstance(item, torch.Tensor)]

            if len(tensors_args) == 1:
                self._activations[layer_name] = tensors_args[0]

            elif len(tensors_args) > 1:
                logger.debug(f"[Layer {layer_name}]: Multiple tensor args; using the first tensor.")
                self._activations[layer_name] = tensors_args[0]

            elif len(tensors_kwargs) == 1:
                logger.debug(f"[Layer {layer_name}]: Single tensor in kwargs; using that tensor.")
                self._activations[layer_name] = tensors_kwargs[0]

            elif len(tensors_kwargs) > 1:
                raise ValueError(f"[Layer {layer_name}]: Multiple tensor kwargs; cannot determine which to use.")

            else:
                raise ValueError(f"[Layer {layer_name}]: Input has no tensor arguments.")

        return hook_fn

    def _register_hooks(self) -> None:
        """Register forward hooks for the specified layers."""
        for layer_name, layer_module in self._layers:
            if self.capture_output:
                handle = layer_module.register_forward_hook(
                    self._create_output_hook(layer_name),
                    prepend=False,
                    with_kwargs=False,
                )
            else:
                handle = layer_module.register_forward_pre_hook(
                    self._create_input_hook(layer_name),
                    prepend=False,
                    with_kwargs=True,
                )

            self._handles.append(handle)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def close(self) -> None:
        """Clean up hooks."""
        self._remove_hooks()

    def __del__(self):
        self.close()


class ActivationLoss(torch.nn.Module):
    def __init__(
        self,
        loss_fn: Callable[..., torch.Tensor],
        reduction: str = "sum-mean",
    ):
        """
        A loss module for aggregating per-layer activation losses into one final scalar.

        This class is useful when a model produces intermediate activations from multiple
        layers and you want to compute a loss term at each layer, then combine (aggregate)
        those layer-wise losses into a single scalar loss.

        Args:
            loss_fn (Callable[..., torch.Tensor, **kwargs]):
                A function that computes the loss for a given layer across input activation
                dictionaries. It is called once per layer.
                - Signature: `loss_fn(*args: torch.Tensor | Any, **kwargs) -> torch.Tensor`
                - It must accept one or more Tensors (corresponding to the same layer key
                  across multiple input dictionaries) and optionally additional kwargs,
                  and return a Tensor of shape `(batch_size,)`,
                  representing the per-sample loss for that layer.

            reduction (str):
                A reduction function which aggregates per-layer losses into a single scalar.
                A tensor of shape `(batch_size, num_layers)` is reduced to a single scalar.
                Supported values are:
                - "sum-mean": Sum over layers, then mean over batch (default).
                - "mean-sum": Mean over layers, then sum over batch.
                - "sum": Sum over all elements.
                - "mean": Mean over all elements.
                - "none": No reduction; returns the full per-layer loss tensor of shape `(batch_size, num_layers)`.
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.reduction = reduction

    def forward(self, *args: dict[str, Any], **kwargs) -> torch.Tensor:
        """
        Args:
            *args: One or more dictionaries of layer activations and/or other data.
                Each dictionary must have the same set of keys (layer names).
            **kwargs: Additional keyword arguments to pass to the loss function.

        Returns:
            torch.Tensor: The aggregated loss as a scalar tensor of shape (1,).
        """
        keys = list(args[0].keys())

        key0 = keys[0]
        loss = self.loss_fn(*[arg[key0] for arg in args], **kwargs)
        losses = torch.zeros((loss.size(0), len(keys)), device=loss.device, dtype=loss.dtype)
        losses[:, 0] = loss

        for i, key in enumerate(keys[1:], start=1):
            loss = self.loss_fn(*[arg[key] for arg in args], **kwargs)
            losses[:, i] = loss

        return self.call_reduction(losses)

    def call_reduction(self, losses: Tensor) -> Tensor:        
        if self.reduction == "sum-mean":
            return losses.sum(dim=-1).mean()
        if self.reduction == "mean-sum":
            return losses.mean(dim=-1).sum()
        if self.reduction == "sum":
            return losses.sum()
        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "none":
            return losses

        raise ValueError(f"Unknown reduction method: {self.reduction}")
