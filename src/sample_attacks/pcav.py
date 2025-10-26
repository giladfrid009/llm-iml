from __future__ import annotations
import inspect
from typing import Callable, Iterable
from tqdm.auto import tqdm
import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression

from src.adv_model import AdvModel
from src.sample_attacks.base import SampleAttack, SampleOutput
from src.activ_extractor import ActivationExtractor, ActivationLoss
from src.initialize import Initializer
from src.aliases import Conv


class RefusalClassifier(torch.nn.Module):
    """
    A logistic regression refusal classifier operating on model activations.
    """

    def __init__(self, w: torch.Tensor, b: torch.Tensor, acc: float = 1.0):
        """
        Args:
            w (torch.Tensor): Weights tensor of shape (hidden_size,).
            b (torch.Tensor): Bias tensor (scalar).
            acc (float): Accuracy of the classifier on evaluation data.
        """
        super().__init__()
        assert w.dim() == 1  # (hidden_size,)
        assert b.dim() == 0  # scalar bias

        self.w = w
        self.b = b
        self.acc = acc

    @classmethod
    def fit(
        cls,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_eval: torch.Tensor,
        y_eval: torch.Tensor,
        max_iter: int = 10000,
    ) -> RefusalClassifier:
        """
        Fit a logistic regression refusal classifier.

        *Note:* the evaluation set is used to compute the accuracy of the classifier.

        Args:
            x_train (torch.Tensor): Training samples of shape (num_train, hidden_size).
            y_train (torch.Tensor): Training labels of shape (num_train,).
            x_eval (torch.Tensor): Evaluation samples of shape (num_eval, hidden_size).
            y_eval (torch.Tensor): Evaluation labels of shape (num_eval,).
            max_iter (int): Maximum number of iterations for the logistic regression solver.

        Returns:
            RefusalClassifier: A fitted refusal classifier.

        """
        # fit logistic regression
        solver = LogisticRegression(solver="saga", max_iter=max_iter)
        solver.fit(x_train.numpy(force=True), y_train.numpy(force=True))

        # extract weights, bias, and accuracy
        w = torch.tensor(torch.tensor(solver.coef_)).squeeze()
        b = torch.tensor(torch.tensor(solver.intercept_)).squeeze()
        acc = solver.score(x_eval.numpy(force=True), y_eval.numpy(force=True))
        return cls(w, b, acc=float(acc))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, hidden_size).

        Returns:
            torch.Tensor: Refusal probability tensor of shape (batch_size,).
        """
        logits = self.logits(x)
        return torch.sigmoid(logits)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, hidden_size).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size,).
        """
        return torch.matmul(x, self.w) + self.b


def refusal_loss(
    activs: torch.Tensor,
    clf: RefusalClassifier,
) -> torch.Tensor:
    """
    Compute the binary cross-entropy loss for refusal classification.
    The loss encourages the model to produce activations that are classified as non-refusal (label 0).

    Args:
        activs (torch.Tensor): The activations from a specific layer, of shape (batch_size, sequence_length, hidden_size).
        clf (RefusalClassifier): The refusal classifier.

    Returns:
        torch.Tensor: The binary cross-entropy loss for every sample, of shape (batch_size,).
    """
    last_activ = activs[:, -1]  # (B, H)
    logits = clf.logits(last_activ)  # (B,)
    target = torch.zeros_like(logits)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return loss


class PCAV(SampleAttack):
    """
    Perturb Concept Activation Vectors (PCAV) Attack.
    Inspired from SCAV [https://arxiv.org/pdf/2404.12038]
    """

    def __init__(
        self,
        adv_model: AdvModel,
        classifiers: dict[str, RefusalClassifier],
        optim_factory: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer],
        steps: int = 100,
        min_acc: float = 0.9,
        noise_scale: float = 0.0,
        mixed_precision: bool = False,
        verbose: bool = True,
    ):
        """
        Args:
            adv_model (AdvModel): The adversarial model to attack.
            optim_factory (Callable[Iterable[torch.Tensor], torch.optim.Optimizer]):
                A factory function that creates an optimizer given the parameters to optimize.
            classifiers (dict[str, RefusalClassifier]): A dictionary mapping layer names to refusal classifiers.
            steps (int): Number of optimization steps.
            min_acc (float): Minimum accuracy of the refusal classifiers to be used.
            noise_scale (float): Standard deviation of Gaussian noise added to the initial embeddings.
            mixed_precision (bool): Whether to use mixed precision training.
            verbose (bool): Whether to display a progress bar.
        """
        super().__init__(adv_model, verbose)

        self.steps = steps
        self.optim_factory = optim_factory
        self.noise_scale = noise_scale
        self.mixed_precision = mixed_precision

        self.min_acc = min_acc
        self.classifiers = {name: clf.to(self.device) for name, clf in classifiers.items() if clf.acc >= min_acc}
        self.extractor = ActivationExtractor(adv_model.model, *list(self.classifiers.keys()))

        if len(self.classifiers) == 0:
            raise ValueError(f"No classifiers with accuracy >= {min_acc}")

    def get_hparams(self) -> dict:
        dummy_optim = self.optim_factory([torch.zeros(1)])
        return {
            "name": self.__class__.__name__,
            "steps": self.steps,
            "layers": list(self.classifiers.keys()),
            "min_acc": self.min_acc,
            "mixed_precision": self.mixed_precision,
            "noise_scale": self.noise_scale,
            "optim": dummy_optim.state_dict()["param_groups"][0],
            "optim/name": dummy_optim.__class__.__name__,
            "optim_factory": inspect.getsource(self.optim_factory),
        }

    def _create_embeddings(
        self,
        num_inputs: int,
        init_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if init_embeds is not None:
            init_embeds = init_embeds.clone().detach()
        else:
            init_embeds = Initializer.random_normal(self.adv_model, std=0.1, batch_size=num_inputs)

        if self.noise_scale > 0.0:
            noise = torch.randn_like(init_embeds) * self.noise_scale
            init_embeds = init_embeds + noise

        init_embeds = init_embeds.contiguous().requires_grad_(True)
        return init_embeds

    def fit(
        self,
        conversations: list[Conv],
        target_texts: list[str],
        init_embeds: torch.Tensor | None = None,
    ) -> SampleOutput:
        # initialize optimized embeddings
        adv_embeds = self._create_embeddings(
            num_inputs=len(conversations),
            init_embeds=init_embeds,
        )

        # create optimizer and scaler
        scaler = torch.GradScaler(enabled=self.mixed_precision)
        optim = self.optim_factory([adv_embeds])

        # tokenize
        conversations = self.adv_model.inject_tokens(conversations)
        encodings = self.adv_model.tokenize(conversations)

        with tqdm(range(self.steps), disable=not self.verbose, leave=False, desc="Attack") as pbar:
            for step in pbar:
                optim.zero_grad()

                with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                    # forward pass
                    with self.extractor.capture():
                        self.adv_model.forward(
                            input_ids=encodings.input_ids,
                            attention_mask=encodings.attention_mask,
                            adv_mask=encodings.adv_mask,
                            adv_embeds=adv_embeds,
                        )

                    activs = self.extractor.get_activations()
                    criterion = ActivationLoss(loss_fn=refusal_loss, aggr_fn=torch.sum)
                    loss = criterion.forward(activs)

                # backward pass and optimization step
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                # update progress bar
                pbar.set_postfix(loss=loss.item())

        return SampleOutput(conversations, adv_embeds.detach())
