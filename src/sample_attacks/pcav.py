from __future__ import annotations
import inspect
from typing import Callable, Iterable
from tqdm.auto import tqdm
import copy

from src.data import TableLoader
from src.adv_model import AdvModel
from src.sample_attacks.base import SampleAttack, SampleOutput
from src.activ_extractor import ActivationExtractor, ActivationLoss
from src.initialize import Initializer
from src.aliases import Conv
from src.utils.logging import create_logger

import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from transformers.tokenization_utils_base import BatchEncoding
from transformers.cache_utils import DynamicCache


logger = create_logger(__name__)


# POTENTIAL DATASETS:
# allenai/wildguardmix
# PKU-Alignment/BeaverTails

# NOTE: we can use Nvidia cuML instead of sklearn for GPU acceleration if needed
# https://github.com/rapidsai/cuml
# uv add "cuml-cu12==25.10.*"


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

        self.w = torch.nn.Parameter(w, requires_grad=False)
        self.b = torch.nn.Parameter(b, requires_grad=False)
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
        # prepare data
        x_train, y_train = train_data
        x_eval, y_eval = eval_data
        x_train = x_train.float().cpu()
        x_eval = x_eval.float().cpu()
        y_train = y_train.int().cpu()
        y_eval = y_eval.int().cpu()

        # fit logistic regression
        solver = LogisticRegression(solver="saga", max_iter=max_iter)
        solver.fit(x_train.numpy(force=True), y_train.numpy(force=True))

        # create model
        w = torch.tensor(solver.coef_).squeeze()
        b = torch.tensor(solver.intercept_).squeeze()
        clf = cls(w, b)

        # compute accuracy
        acc = (clf.predict(x_eval) == y_eval).float().mean().item()
        clf.acc = acc

        return clf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)

    def proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the probability of the positive class.
        """
        logits = self.logits(x)
        return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels (0 or 1) based on the input activations.
        """
        return self.proba(x).round()

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the logits for the positive class.
        """
        x = x.to(self.w.dtype)
        return torch.matmul(x, self.w) + self.b


class LogisticTrainer:
    """
    Trainer for logistic regression models.
    Extracts last token activations from specified layers and fits logistic regression models to classify refusals.
    """

    def __init__(
        self,
        adv_model: AdvModel,
        activ_extractor: ActivationExtractor,
        mixed_precision: bool = False,
        verbose: bool = True,
    ):
        """
        Args:
            adv_model (AdvModel): The attack target model.
            activ_extractor (ActivationExtractor): Extractor for capturing activations from specified layers.
            mixed_precision (bool): Whether to use mixed precision training.
            verbose (bool): Whether to display progress bars.
        """
        self.adv_model = adv_model
        self.extractor = activ_extractor
        self.mixed_precision = mixed_precision
        self.verbose = verbose

    def get_hparams(self) -> dict:
        return {
            "activ_extractor": self.extractor.get_hparams(),
            "mixed_precision": self.mixed_precision,
            "verbose": self.verbose,
        }

    def extract_activations(self, dl: TableLoader) -> dict[str, torch.Tensor]:
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

        for batch in tqdm(dl, disable=not self.verbose, desc="Extracting Activations", leave=False):
            conversations = [[{"role": "user", "content": text}] for text in batch["prompt"]]
            encodings = self.adv_model.tokenize(conversations)

            with (
                torch.autocast(device_type=self.adv_model.device.type, enabled=self.mixed_precision),
                torch.inference_mode(),
            ):
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
            (dict[str, LogisticModel]): A dictionary mapping layer names to fitted refusal classifiers.
        """
        dl_train.validate(["prompt", "label"])
        dl_eval.validate(["prompt", "label"])

        if dl_train.shuffle or dl_eval.shuffle:
            raise ValueError("DataLoaders for training/evaluation should not be shuffled.")

        logger.info(f"dl_train info: {dl_train.get_hparams()}")
        logger.info(f"dl_eval info: {dl_eval.get_hparams()}")

        classifiers: dict[str, LogisticModel] = {}
        activs_train = self.extract_activations(dl_train)
        activs_eval = self.extract_activations(dl_eval)

        for layer_name in tqdm(
            self.extractor.layer_names,
            disable=not self.verbose,
            desc="Training Logistic-Regression Models",
            unit="layer",
            leave=False,
        ):
            # prepare data
            train_x = activs_train[layer_name]
            train_y = torch.tensor(dl_train.df["label"].to_numpy(dtype="int32"))

            eval_x = activs_eval[layer_name]
            eval_y = torch.tensor(dl_eval.df["label"].to_numpy(dtype="int32"))

            # fit model
            clf = LogisticModel.fit(train_data=(train_x, train_y), eval_data=(eval_x, eval_y), **kwargs)
            classifiers[layer_name] = clf

        if self.verbose:
            logger.info("Trained CAV Classifiers:")
            for layer_name, clf in classifiers.items():
                logger.info(f"Layer: {layer_name}, Features: {clf.w.size(0)}, Accuracy: {clf.acc}")

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
        activ_extractor: ActivationExtractor,
        optim_factory: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer],
        steps: int = 100,
        min_acc: float = 0.9,
        target_prob: float = 0.1,
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
            activ_extractor (ActivationExtractor): Extractor for capturing activations.
            steps (int): Number of optimization steps.
            min_acc (float): Minimum accuracy of the logistic models to be used.
            target_prob (float): Target probability threshold for early stopping.
            noise_scale (float): Standard deviation of Gaussian noise added to the initial embeddings.
            mixed_precision (bool): Whether to use mixed precision training.
            verbose (bool): Whether to display a progress bar.
        """
        super().__init__(adv_model, verbose)

        cls_layers = set(classifiers.keys())
        act_layers = set(activ_extractor.layer_names)

        if not cls_layers.issubset(act_layers):
            missing = cls_layers - act_layers
            raise ValueError(f"Activation extractor is missing layers required by classifiers: {missing}")

        elif not act_layers.issubset(cls_layers):
            extra = act_layers - cls_layers
            logger.warning(f"Activation extractor has extra layers not used by classifiers: {extra}")

        # filter classifiers with low accuracy
        classifiers = {name: clf for name, clf in classifiers.items() if clf.acc >= min_acc}
        classifiers = {name: clf.to(self.device) for name, clf in classifiers.items()}

        if len(classifiers) == 0:
            raise ValueError(f"No classifiers with accuracy >= {min_acc}")

        # create activation extractor only for layers with classifiers
        activ_extractor = ActivationExtractor(
            activ_extractor.model,
            *[l for l in activ_extractor.layer_names if l in classifiers.keys()],
            exact_match=activ_extractor.exact_match,
            capture_output=activ_extractor.capture_output,
        )

        self.optim_factory = optim_factory
        self.classifiers = classifiers
        self.extractor = activ_extractor
        self.steps = steps
        self.min_acc = min_acc
        self.target_prob = target_prob
        self.noise_scale = noise_scale
        self.mixed_precision = mixed_precision

    @property
    def layers(self) -> list[str]:
        return list(self.classifiers.keys())

    def get_hparams(self) -> dict:
        dummy_optim = self.optim_factory([torch.zeros(1)])
        return {
            "name": type(self).__name__,
            "steps": self.steps,
            "layers": self.layers,
            "min_acc": self.min_acc,
            "target_prob": self.target_prob,
            "mixed_precision": self.mixed_precision,
            "noise_scale": self.noise_scale,
            "optim": dummy_optim.state_dict()["param_groups"][0],
            "optim/name": type(dummy_optim).__name__,
            "optim_factory": inspect.getsource(self.optim_factory),
            "extractor": self.extractor.get_hparams(),
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

    @torch.no_grad()
    def _compute_cache(self, encodings: BatchEncoding) -> BatchEncoding:
        """
        Compute the kv-cache for the constant part of the input.

        Args:
            encodings (BatchEncoding): encodings containing input tokens and masks,
                as returned from `self.adv_model.tokenize()`.

        Returns:
            BatchEncoding: encodings containing the kv-cache for the constant part of the input,
                as well as the remaining input tokens and masks. The remaining input tokens
                and masks are only the variable (non-cached) part of the input.
        """

        kv_idx = encodings.const_idx.min().item()

        kv_result = self.adv_model.forward(
            encodings.input_ids[:, :kv_idx],
            encodings.attention_mask[:, :kv_idx],
            adv_mask=None,
            adv_embeds=None,
            use_cache=True,
        )

        kv_cache = kv_result.past_key_values
        if isinstance(kv_cache, tuple):  # convert legacy cache format
            kv_cache = DynamicCache.from_legacy_cache(kv_cache)

        new_data = {
            "input_ids": encodings.input_ids[:, kv_idx:],
            "attention_mask": encodings.attention_mask,  # we need the full attention mask
            "adv_mask": encodings.adv_mask[:, kv_idx:],
            "kv_cache": kv_cache,
        }

        if "target_mask" in encodings:
            new_data["target_mask"] = encodings.target_mask[:, kv_idx:]

        return BatchEncoding(new_data)

    def _masked_select(
        self,
        encodings: BatchEncoding,
        mask: torch.Tensor,
    ) -> BatchEncoding:
        """
        Selects elements from the encodings based on the provided mask.
        Mask is applied to the batch dimension of all tensor elements in the encodings.
        """
        new_data = {}
        for key, value in encodings.data.items():
            if isinstance(value, torch.Tensor):
                new_data[key] = value[mask]

            elif key == "kv_cache" and value is None:
                new_data[key] = None

            elif key == "kv_cache" and isinstance(value, DynamicCache):
                indices = mask.nonzero(as_tuple=True)[0]
                cache = value.batch_select_indices(indices)
                new_data[key] = cache

            else:
                raise TypeError(f"Unsupported type {type(value)} for key {key} in encodings")

        return BatchEncoding(new_data)

    @torch.inference_mode()
    def _check_early_stopping(self, activs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Check which samples have met the early stopping criteria based on the classifiers' predictions.
        If all classifiers predict non-refusal (probability <= target_prob) for a sample,
        that sample is considered finished.

        Args:
            activs (dict[str, torch.Tensor]): A dictionary mapping layer names to activation tensors of
                shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: A boolean tensor of shape (batch_size,) indicating which
                samples have met the early stopping criteria.
        """
        finished_status = torch.ones(
            activs[next(iter(activs))].size(0),
            dtype=torch.bool,
            device=self.device,
        )

        for layer_name, clf in self.classifiers.items():
            probs = clf.proba(activs[layer_name][:, -1])  # (B,)
            status = probs <= self.target_prob
            finished_status = finished_status & status

        return finished_status

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
        criterion = ActivationLoss(loss_fn=refusal_loss, reduction="sum")

        # tokenize
        conversations = self.adv_model.inject_tokens(conversations)
        encodings = self.adv_model.tokenize(conversations)

        # compute kv-cache
        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
            encodings = self._compute_cache(encodings)

        # early stopping state
        finished = torch.zeros(len(conversations), dtype=torch.bool, device=self.device)
        optim_embeds = adv_embeds

        LOGS = {"loss": [], "remaining": []}

        with tqdm(range(self.steps), disable=not self.verbose, leave=False, desc="Attack") as pbar:
            for step in pbar:
                optim.zero_grad()

                # NOTE: need to copy kv-cache since forward modifies it in-place
                step_encodings = encodings.copy()
                step_encodings["kv_cache"] = copy.deepcopy(encodings.kv_cache)

                # select only unfinished samples if early stopping is enabled
                if self.target_prob > 0.0:
                    step_encodings = self._masked_select(step_encodings, ~finished)
                    optim_embeds = adv_embeds[~finished]

                with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                    # forward pass
                    with self.extractor.capture():
                        self.adv_model.forward(
                            input_ids=step_encodings.input_ids,
                            attention_mask=step_encodings.attention_mask,
                            adv_mask=step_encodings.adv_mask,
                            past_key_values=step_encodings.kv_cache,
                            adv_embeds=optim_embeds,
                        )

                        activs = self.extractor.get_activations()

                    # update early stopping
                    if self.target_prob > 0.0:
                        finished_status = self._check_early_stopping(activs)

                        finished[~finished] = finished_status
                        if finished.all():  # break early
                            pbar.n = pbar.total
                            pbar.close()
                            break

                        activs = {layer: acts[~finished_status] for layer, acts in activs.items()}

                    loss = criterion.forward(activs, self.classifiers, target_value=0)

                # backward pass and optimization step
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                # update progress bar
                num_remaining = (~finished).sum().item()
                remaining = f"{num_remaining}/{len(conversations)}"
                pbar.set_postfix(loss=loss.item(), remaining=remaining)

                LOGS["loss"].append(loss.item() / num_remaining)
                LOGS["remaining"].append(num_remaining)

        return SampleOutput(conversations, adv_embeds.detach(), logs=LOGS)
