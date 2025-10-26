from __future__ import annotations
import inspect
from typing import Callable, Iterable
from tqdm.auto import tqdm
import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression

from src.data import TableLoader
from src.adv_model import AdvModel
from src.sample_attacks.base import SampleAttack, SampleOutput
from src.activ_extractor import ActivationExtractor, ActivationLoss
from src.initialize import Initializer
from src.aliases import Conv


class LogisticModel(torch.nn.Module):
    """
    A logistic regression model operating on model activations.
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
        train_data: tuple[torch.Tensor, torch.Tensor],
        eval_data: tuple[torch.Tensor, torch.Tensor],
        max_iter: int = 10000,
    ) -> LogisticModel:
        """
        Fit a logistic regression model using scikit-learn.

        *Note:* the evaluation set is used to compute the accuracy of the model.

        Args:
            train_data (tuple[torch.Tensor, torch.Tensor]): A tuple of training samples and labels (X, y).
                Shape of samples: (num_samples, hidden_size); shape of labels: (num_samples,).
            eval_data (tuple[torch.Tensor, torch.Tensor]): A tuple of evaluation samples and labels (X, y).
                Shape of samples: (num_samples, hidden_size); shape of labels: (num_samples,).
            max_iter (int): Maximum number of iterations for the logistic regression solver.

        Returns:
            LogisticModel: A fitted instance of the model.
        """
        x_train, y_train = train_data
        x_eval, y_eval = eval_data

        solver = LogisticRegression(solver="saga", max_iter=max_iter)
        solver.fit(x_train.numpy(force=True), y_train.numpy(force=True))
        w = torch.tensor(torch.tensor(solver.coef_)).squeeze()
        b = torch.tensor(torch.tensor(solver.intercept_)).squeeze()
        acc = solver.score(x_eval.numpy(force=True), y_eval.numpy(force=True))
        return cls(w, b, acc=float(acc))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.logits(x)
        return torch.sigmoid(logits)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.w) + self.b


class LogisticTrainer:
    """
    Trainer for logistic regression models.
    Extracts last token activations from specified layers and fits logistic regression models to classify refusals.
    """

    def __init__(
        self,
        adv_model: AdvModel,
        layers: list[str],
        mixed_precision: bool = False,
        verbose: bool = True,
    ):
        """
        Args:
            adv_model (AdvModel): The adversarial model.
            layers (list[str]): List of layer names to extract activations from.
            mixed_precision (bool): Whether to use mixed precision training.
            verbose (bool): Whether to display progress bars.
        """
        self.adv_model = adv_model
        self.mixed_precision = mixed_precision
        self.verbose = verbose
        self.extractor = ActivationExtractor(adv_model.model, *layers, capture_output=True)

    @torch.inference_mode()
    def extract_activations(
        self,
        dl: TableLoader,
    ) -> dict[str, torch.Tensor]:
        """
        Extracts last token activations from the given DataLoader.

        Args:
            dl (TableLoader): DataLoader providing batches of data.

        Returns:
            dict[str, torch.Tensor]: A dictionary mapping layer names to last token activation tensors of shape (num_samples, hidden_size).
        """
        if dl.shuffle:
            raise ValueError("DataLoader for extracting activations should not be shuffled.")

        activs_dict: dict[str, list[torch.Tensor]] = {layer: [] for layer in self.extractor.layer_names}

        for batch in tqdm(dl, disable=not self.verbose, desc="Extracting Activations"):
            conversations = [[{"role": "user", "content": text}] for text in batch["prompt"]]
            encodings = self.adv_model.tokenize(conversations)

            with torch.autocast(device_type=self.adv_model.device.type, enabled=self.mixed_precision):
                with self.extractor.capture():
                    self.adv_model.forward(encodings.input_ids, encodings.attention_mask, adv_mask=None)

                batch_activs = self.extractor.get_activations()
                for layer, acts in batch_activs.items():
                    last_acts = acts[:, -1].cpu().clone().detach()  # (B, H)
                    activs_dict[layer].append(last_acts)

        activs_final: dict[str, torch.Tensor] = {}
        for layer, acts_list in activs_dict.items():
            activs_final[layer] = torch.cat(acts_list, dim=0)
        return activs_final

    @torch.inference_mode()
    def fit(
        self,
        dl_train: TableLoader,
        dl_eval: TableLoader,
        **kwargs,
    ) -> dict[str, LogisticModel]:
        """
        Fit refusal classifiers for each specified layer.

        Args:
            dl_train (TableLoader): DataLoader for training data.
            dl_eval (TableLoader): DataLoader for evaluation data.
            **kwargs: Additional keyword arguments passed to `LogisticModel.fit()`.

        Returns:
            dict[str, LogisticModel]: A dictionary mapping layer names to fitted refusal classifiers.
        """
        dl_train.validate(["prompt", "label"])
        dl_eval.validate(["prompt", "label"])

        if dl_train.shuffle or dl_eval.shuffle:
            raise ValueError("DataLoaders for training/evaluation should not be shuffled.")

        classifiers = {}
        activs_train = self.extract_activations(dl_train)
        activs_eval = self.extract_activations(dl_eval)

        for layer_name in tqdm(
            self.extractor.layer_names,
            disable=not self.verbose,
            desc="Training Logistic Regression Models",
            unit="layer",
        ):
            # prepare data
            train_x = activs_train[layer_name]
            train_y = torch.tensor(dl_train.df["label"], dtype=torch.int32)

            eval_x = activs_eval[layer_name]
            eval_y = torch.tensor(dl_eval.df["label"], dtype=torch.int32)

            # fit model
            clf = LogisticModel.fit(train_data=(train_x, train_y), eval_data=(eval_x, eval_y), **kwargs)
            classifiers[layer_name] = clf

        return classifiers


def refusal_loss(activs: torch.Tensor, clf: LogisticModel, target_value: int = 0) -> torch.Tensor:
    """
    Compute the BCE loss for refusal classification.
    The loss encourages the logistic model to produce activations that are classified as the target value.

    Args:
        activs (torch.Tensor): Activations tensor of shape (batch_size, seq_len, hidden_size).
        clf (LogisticModel): The logistic regression classifier.
        target_value (int): The target class (0 or 1).

    Returns:
        torch.Tensor: The computed loss tensor of shape (batch_size,).
    """
    last_activ = activs[:, -1]  # (B, H)
    logits = clf.logits(last_activ)  # (B,)
    target = torch.tensor(target_value, dtype=logits.dtype, device=logits.device).expand_as(logits)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return loss


class PCAV(SampleAttack):
    """
    Perturb Concept Activation Vectors (P-CAV) Attack.
    Inspired from SCAV [https://arxiv.org/pdf/2404.12038]
    """

    def __init__(
        self,
        adv_model: AdvModel,
        classifiers: dict[str, LogisticModel],
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
            classifiers (dict[str, LogisticModel]): A dictionary mapping layer names to logistic models.
            steps (int): Number of optimization steps.
            min_acc (float): Minimum accuracy of the logistic models to be used.
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
