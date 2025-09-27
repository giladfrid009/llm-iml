import torch
import torch.optim


class GradSign(torch.optim.Optimizer):
    """
    A PyTorch optimizer that applies sign to the gradients of the model's parameters before applying the internal optimizer step.
    This optimizer is intended to be used as a wrapper around an existing optimizer.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        super().__init__(optimizer.param_groups, optimizer.defaults)
        self.optimizer = optimizer
        self.state = optimizer.state

    def step(self, closure=None) -> float | None:  # type: ignore
        # Normalizes all gradients to be sign vectors
        # Before applying the internal optimizer step

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.grad.sign_()

        return self.optimizer.step(closure)


class FGSM(GradSign):
    def __init__(
        self,
        params,
        lr=0.001,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        maximize=False,
        foreach=None,
        differentiable=False,
        fused=None,
        **kwargs,
    ):
        optimizer = torch.optim.SGD(
            params=params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
            **kwargs,
        )

        super().__init__(optimizer=optimizer)
