from src.attacks.attack import Attack
from src.adver_model import AdverModel
from src.activ_extractor import ActivationExtractor, ActivationLoss
from src.data import DF_Batcher
from src.eval.evaluator import Evaluator
from src.logger import Logger

from typing import Any
import torch
from tqdm.auto import tqdm
import time
from typing import Iterable, Callable
import warnings
import pathlib
from functools import partial


class StopCriteria:
    def __init__(
        self,
        max_epochs: int = 100,
        max_evals: int | None = None,
        max_time: float | None = None,
        target_value: float | None = None,
        patience: int | None = None,
        patience_delta: float = 1e-4,
    ):
        """
        Container for various stopping criteria.

        Args:
            max_epochs: Maximum number of epochs.
            max_evals: Maximum number of evaluation steps.
            max_time: Maximum training time in seconds.
            target_value: Target value to stop training when reached. This value should be maximized.
            patience: Number of evaluation steps without sufficient improvement.
            patience_delta: Minimum improvement delta to reset patience.

        Raises:
            ValueError: If any of the arguments are invalid.
        """
        if max_epochs <= 0:
            raise ValueError("Max epochs must be greater than 0.")
        if max_evals is not None and max_evals <= 0:
            raise ValueError("Max evals must be greater than 0.")
        if max_time is not None and max_time <= 0:
            raise ValueError("Max time must be greater than 0.")
        if patience is not None and patience <= 0:
            raise ValueError("Patience must be greater than 0.")
        if patience_delta < 0:
            raise ValueError("Patience delta must be greater than or equal to 0.")

        self.max_epochs = max_epochs
        self.max_evals = max_evals
        self.max_time = max_time
        self.target_value = target_value
        self.patience = patience
        self.patience_delta = patience_delta

        # Internal state
        self._epoch = 0
        self._total_evals = 0
        self._start_time = time.time()
        self._best_value = -float("inf")
        self._patience_counter = 0
        self.reset()

    def get_hparams(self) -> dict:
        return {
            "stop/max_epochs": self.max_epochs,
            "stop/max_evals": self.max_evals,
            "stop/max_time": self.max_time,
            "stop/target_value": self.target_value,
            "stop/patience": self.patience,
            "stop/patience_delta": self.patience_delta,
        }

    def reset(self) -> None:
        """Reset internal state."""
        self._epoch = 0
        self._total_evals = 0
        self._start_time = time.time()
        self._best_value = -float("inf")
        self._patience_counter = 0

    def update(self, epoch: int, value: float | None) -> None:
        """Update internal state with new metrics."""
        self._epoch = epoch
        self._total_evals += 1

        if value is not None:
            if (value - self._best_value) >= self.patience_delta:
                self._best_value = value
                self._patience_counter = 0
            else:
                self._patience_counter += 1

    def should_stop(self) -> bool:
        """Check if any stopping condition is met."""
        if self.target_value is not None and self._best_value >= self.target_value:
            print(f"Stopping: Target value reached :: ({self.target_value})")
            return True

        if self._epoch >= self.max_epochs:
            print(f"Stopping: Max epochs reached :: ({self.max_epochs})")
            return True

        if self.max_evals is not None and self._total_evals >= self.max_evals:
            print(f"Stopping: Max evals reached :: ({self.max_evals})")
            return True

        if self.patience is not None and self._patience_counter >= self.patience:
            print(f"Stopping: Patience exceeded :: ({self.patience})")
            return True

        if self.max_time is not None and (time.time() - self._start_time) > self.max_time:
            print(f"Stopping: Max time reached :: ({self.max_time} sec)")
            return True

        return False


# NOTE: currently loss is over all target tokens and not a single token per sample
def cosine_similarity_loss(univ_activ: torch.Tensor, sample_activ: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
        univ_activ (torch.Tensor): Universal activations of shape (batch_size, seq_length, hidden_dim).
        sample_activ (torch.Tensor): Sample activations of shape (batch_size, seq_length, hidden_dim).
        target_mask (torch.Tensor): Mask indicating which tokens are targets of shape (batch_size, seq_length).
    """
    cos_sim = torch.cosine_similarity(univ_activ, sample_activ, dim=-1)  # (batch_size, seq_length)
    loss_matrix = (1 - cos_sim) * target_mask.bool()  # (batch_size, seq_length)
    loss = torch.mean(loss_matrix.sum(dim=-1) / target_mask.sum(dim=-1))
    return loss


# TODO: add scheduling to the inner-attack i.e accept a lambda function that takes the current epoch and
# returns an instance of an inner attack.
# TODO: implmenet discretization
class IML_Attack:
    def __init__(
        self,
        adv_model: AdverModel,
        internal_attack: Attack,
        optim_factory: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer],
        activ_extractor: ActivationExtractor,
        evaluators: list[Evaluator],
        eval_freq: int | float = 1,
        mixed_precision: bool = True,
        pred_kwargs: dict[str, Any] | None = None,
        skip_already_fooled: bool = False,
        skip_failed_attacks: bool = True,
        log_dir: str | None = None,
    ):
        """
        Args:
            adv_model (AdverModel): The adversarial model to use for text generation.
            internal_attack (Attack): The internal attack to use for generating adversarial examples.
            optim_factory (Callable): Function to create an optimizer for the universal embeddings.
            activ_extractor (ActivationExtractor): Activation extractor to capture model activations
                from latent layers. Loss computation is based on these activations.
            evaluators (list[Evaluator]): List of evaluators to use for evaluation.
            eval_freq (int | float): Frequency of evaluation during training.
                - if int, evaluates every `eval_freq` epochs.
                - if float, evaluates every `round(eval_freq * len(dl_train))` batches.
            mixed_precision (bool): Whether to use mixed precision training.
            pred_kwargs (dict[str, Any] | None): Additional keyword arguments for `AdverModel.chat`.
            skip_already_fooled (bool): If True, skips samples that are already successfully fooled.
            skip_failed_attacks (bool): If True, skips samples where the internal attack fails.
            log_dir (str | None): Directory to save logs. If None, no logging is performed.
        """
        self.adv_model = adv_model
        self.internal_attack = internal_attack
        self.activ_extractor = activ_extractor

        self.evaluators = evaluators
        self.eval_freq = eval_freq
        self.pred_kwargs = pred_kwargs if pred_kwargs is not None else {}

        self.mixed_precision = mixed_precision
        self.grad_scaler = torch.GradScaler(enabled=mixed_precision)

        self.univ_embeds = self.adv_model.get_embeddings(clone=False)
        self.univ_embeds.requires_grad_(True)
        if self.univ_embeds.size(0) != 1:
            raise ValueError("Batch size of universal embeddings must be 1.")

        self.optimizer = optim_factory([self.univ_embeds])
        self.skip_already_fooled = skip_already_fooled
        self.skip_failed_attacks = skip_failed_attacks

        self.best_metric = -float("inf")
        self.best_embeds = self.univ_embeds.clone().detach()

        # logging
        self.logger = Logger(log_dir)
        self.logger.register_hparams(self.get_hparams())
        self.logger.register_hparams(adv_model.get_hparams())
        for evaler in self.evaluators:
            self.logger.register_hparams(evaler.get_hparams())
        self.logger.register_hparams({f"internal_attack/{k}": v for k, v in internal_attack.__dict__.items()})
        self.logger.register_hparams({f"optim/name": self.optimizer.__class__.__name__})
        self.logger.register_hparams({f"optim/{k}": v for k, v in self.optimizer.param_groups[0].items()})
        self.logger.register_hparams({f"grad_scaler/{k}": v for k, v in self.grad_scaler.state_dict().items()})

    @property
    def num_tokens(self) -> int:
        return self.adv_model.num_tokens

    @property
    def embed_dim(self) -> int:
        return self.adv_model.adv_embedder.embed_dim

    @property
    def device(self) -> torch.device:
        return self.adv_model.device

    @property
    def judge(self) -> Evaluator:
        """
        Returns the main judge evaluator.
        This is the first evaluator in the list of evaluators.
        """
        return self.evaluators[0]

    def get_hparams(self) -> dict:
        return {
            "iml_attack/num_tokens": self.num_tokens,
            "iml_attack/embed_dim": self.embed_dim,
            "iml_attack/mixed_precision": self.mixed_precision,
            "iml_attack/skip_already_fooled": self.skip_already_fooled,
            "iml_attack/skip_failed_attacks": self.skip_failed_attacks,
            "iml_attack/optimizer": self.optimizer.__class__.__name__,
            "iml_attack/internal_attack": self.internal_attack.__class__.__name__,
            "iml_attack/eval_freq": self.eval_freq,
            "iml_attack/pred_kwargs": self.pred_kwargs,
            "iml_attack/evaluators": (e.name for e in self.evaluators),
            "iml_attack/log_dir": self.logger.root_dir if self.logger else None,
        }

    def close(self):
        """Close all resources."""
        self.logger.close()

    def __del__(self):
        self.close()

    def save_checkpoint(self, file_name: str = "best_embeds.pt"):
        log_dir = self.logger.log_dir()
        if log_dir is not None:
            torch.save(self.best_embeds, pathlib.Path(log_dir) / file_name)

    @torch.inference_mode()
    def predict(
        self,
        adv_model: AdverModel,
        dl_eval: DF_Batcher,
        **kwargs: Any,
    ) -> list[str]:
        """
        Generates model responses for the evaluation data loader.

        Args:
            adv_model (AdverModel): The adversarial model to use for text generation.
            dl_eval (DF_Batcher): Data loader for evaluation.
            **kwargs (dict): Additional keyword arguments for `AdverModel.chat`.

        Returns:
            list[str]: List of generated responses.
        """

        kwargs = kwargs if kwargs else self.pred_kwargs
        dl_eval = dl_eval.copy(shuffle=False, drop_last=False)

        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):

            all_responses = []
            for batch_data in tqdm(dl_eval, desc="Generating", leave=False):
                prompts = batch_data["prompt"]
                conversations = [[{"role": "user", "content": prm}] for prm in prompts]
                responses = adv_model.chat(conversations, **kwargs)
                all_responses.extend(responses)

        return all_responses

    @torch.inference_mode()
    def evaluate(
        self,
        adv_model: AdverModel,
        evalers: list[Evaluator] | Evaluator,
        dl_eval: DF_Batcher,
        update_best: bool = False,
        **kwargs: Any,
    ) -> list[float]:
        """
        Evaluates the model using the provided evaluators and data loader.

        Args:
            adv_model (AdverModel): The adversarial model to evaluate.
            evalers (list[Evaluator]): List of evaluators to use for evaluation.
            dl_eval (DF_Batcher): Data loader for evaluation.
            update_best (bool): If True, updates the best metric and embeddings if the current evaluation is better.
            **kwargs (dict): Additional keyword arguments for `AdverModel.chat`.

        Returns:
            list[float]: List containing evaluation metrics from each evaluator.
        """

        if isinstance(evalers, Evaluator):
            evalers = [evalers]

        # TODO: (low priority) add support to evaluating mutiple generations per prompt
        # easiest and probably cleanest solution is to copy each row in dl_eval multiple times

        if dl_eval.drop_last or dl_eval.shuffle:
            warnings.warn(
                "Evaluation data loader should not be shuffled or dropped last. "
                "Creatomg a shallow copy with `shuffle=False` and `drop_last=False`."
            )
            dl_eval = dl_eval.copy(shuffle=False, drop_last=False)

        all_responses = self.predict(adv_model, dl_eval, **kwargs)
        dl_eval.set_column("response", all_responses)

        metrics = []
        for evaluator in evalers:
            res = evaluator.evaluate(dl_eval)
            metrics.append(res)

        # Find metric which belongs to the judge evaluator
        # If it improves, update the best metric and embeddings
        if update_best:

            judge_metric = None
            for evaler, metric in zip(evalers, metrics):
                if evaler == self.judge:
                    judge_metric = metric
                    break

            if judge_metric is None:
                raise ValueError(f"Judge evaluator {self.judge.name} not found in the list of evaluators {[e.name for e in evalers]}.")

            if self.best_metric < judge_metric:
                self.best_metric = judge_metric
                self.best_embeds = adv_model.get_embeddings(clone=True)

        return metrics

    def fit(
        self,
        dl_train: DF_Batcher,
        dl_eval: DF_Batcher | None = None,
        stop_criteria: StopCriteria | None = None,
    ) -> AdverModel:

        if dl_eval is None:
            warnings.warn("No evaluation data loader provided, using training data for evaluation.")
            dl_eval = dl_train.copy(shuffle=False, drop_last=False)

        # must have these columns at least
        dl_train.validate(["prompt", "target"])

        if stop_criteria is None:
            stop_criteria = StopCriteria()

        # local stats
        global_step = 0
        loss_value = None
        stop_criteria.reset()
        should_stop = stop_criteria.should_stop()

        # init logging
        self.logger.register_hparams(stop_criteria.get_hparams())
        self.logger.initialize(
            self.adv_model.model.name_or_path,
            f"num_tokens_{self.num_tokens}",
            self.internal_attack.__class__.__name__,
        )
        self.logger.add_tags(
            model=self.adv_model.model.name_or_path,
            num_tokens=self.num_tokens,
            internal_attack=self.internal_attack.__class__.__name__,
        )
        self.logger.log_hparams()

        with tqdm(range(stop_criteria.max_epochs), desc="Epochs") as epoch_pbar:

            # initial evaluation
            self.adv_model.set_embeddings(self.univ_embeds)
            metrics = self.evaluate(self.adv_model, self.evaluators, dl_eval, update_best=True)

            self.save_checkpoint()
            self.logger.log_scalar(f"{self.judge.name}/best", self.best_metric, step=-1)
            self.logger.log_scalers({f"{e.name}/current": m for e, m in zip(self.evaluators, metrics)}, step=-1)
            epoch_pbar.set_postfix({e.name: m for e, m in zip(self.evaluators, metrics)})

            # main training loop
            for epoch_num in epoch_pbar:
                if should_stop:
                    break

                with tqdm(dl_train, desc="Batches", leave=False) as batch_pbar:
                    for batch_num, batch_data in enumerate(batch_pbar):
                        if should_stop:
                            break

                        # training step
                        loss_value = self.optim_step(batch_data, epoch_num, batch_num)
                        stop_criteria.update(epoch_num, None)
                        self.logger.log_scalar("loss", loss_value, step=global_step)
                        batch_pbar.set_postfix({"loss": loss_value})
                        should_stop = stop_criteria.should_stop()

                        # evaluation step
                        if should_stop or (global_step > 0 and global_step % round(self.eval_freq * len(dl_train)) == 0):
                            self.adv_model.set_embeddings(self.univ_embeds)
                            metrics = self.evaluate(self.adv_model, self.evaluators, dl_eval, update_best=True)
                            stop_criteria.update(epoch_num, metrics[0])

                            self.save_checkpoint()
                            self.logger.log_scalar(f"{self.judge.name}/best", self.best_metric, step=global_step)
                            self.logger.log_scalers({f"{e.name}/current": m for e, m in zip(self.evaluators, metrics)}, step=global_step)
                            epoch_pbar.set_postfix({e.name: m for e, m in zip(self.evaluators, metrics)})

                        global_step += 1

        # set to best embeddings and final eval
        self.adv_model.set_embeddings(self.best_embeds)
        metrics = self.evaluate(self.adv_model, self.evaluators, dl_eval)
        self.logger.log_metrics({e.name: m for e, m in zip(self.evaluators, metrics)})
        for evaler, value in zip(self.evaluators, metrics):
            print(f"Final metric {evaler.name}: {value:.6f}")

        self.close()
        return self.adv_model

    def optim_step(self, data: dict[str, list[Any]], epoch_num: int, batch_num: int) -> float | None:
        """
        Perform a single optimization step on the given batch of data.

        Args:
            data (dict[str, list[Any]]): Batch data containing input and target texts.
            epoch_num (int): Current epoch number.
            batch_num (int): Current batch number.

        Returns:
            float | None: Loss value for the optimization step, or None if no loss is computed.

        """
        
        # NOTE: IDEA: instead of using a fixed target, generate affirmative responses from the 
        # per-sample attacks and use them as targets instead.
        # We should add attack parameter `dynamic_targets` which enables / disables it.
        # CONS: 
        # 1. very slow since we need to generate responses, but if we use `skip_failed_attacks=True` then 
        # no additional time cost since we generate it anyways.
        # 2. need to be careful with tokenization, tokenize the new generated response as the target
        # PROS: 
        # 1. dynamic targets instead of forced ones
        # 2. will allow to support attacks which do not recieve target argument
        # 3. correctness - `skip_failed_attacks` judges the actual generated responses
        
        # NOTE: IDEA: let the per-sample attacks to also modify the prompt and not only the adversarial tokens.
        # Even more generally - the per-sample attack returns a new adversarial prompt (which may or may not incorporate adv tokens).
        # combined with the previous idea, we then generate an affirmative response to the adver input and use it as the target.
        # CONS:
        # 1. if we allow to modify also the input from the per-sample attack then the affirmative target 
        # might not even correspond to the original prompt, therefore its not clear what we're optimizing in that case
        # 2. in that case the returned outputs should be adversarial embeddings and not adversarial input tokens, since SoftPrompt works on the 
        # embedding level. Therefore we need to support that.
        # PROS:
        # 1. allows unconstrained use of all per-sample attack altogether:
        #   - for example the per-sample attack doesnt have to use the exact number of adver tokens as the universal attack
        #   - new supported attacks:  direct request attack and also human_jailbreaks, and all attacker-LLM based attacks.
        
        
        self.optimizer.zero_grad()

        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):

            # construct conversations
            input_text, target_text = data["prompt"], data["target"]
            conversations = [[{"role": "user", "content": prm}] for prm in input_text]

            # skip already succesfully fooled samples
            if self.skip_already_fooled:
                with torch.inference_mode():
                    self.adv_model.set_embeddings(self.univ_embeds)
                    responses = self.adv_model.chat(conversations, **self.pred_kwargs)
                    data["response"] = responses
                    eval_result = self.judge.process_batch(data)

                    mask_fooled = eval_result >= 1.0
                    if mask_fooled.all():
                        return None

                    conversations = [conv for conv, m in zip(conversations, mask_fooled) if not m]
                    target_text = [tgt for tgt, m in zip(target_text, mask_fooled) if not m]

            # run per-sample attack
            with torch.autocast(device_type=self.device.type, enabled=False):
                init_embeds = self.univ_embeds.expand(len(conversations), -1, -1)
                sample_embed = self.internal_attack.fit(conversations, target_text, init_embeds=init_embeds)

            # skip failed per-sample attacks
            if self.skip_failed_attacks:
                with torch.inference_mode():
                    self.adv_model.set_embeddings(sample_embed)
                    responses = self.adv_model.chat(conversations, **self.pred_kwargs)
                    data["response"] = responses
                    eval_result = self.judge.process_batch(data)

                    mask_succ = eval_result >= 1.0
                    if not mask_succ.any():
                        return None

                    sample_embed = sample_embed[mask_succ]
                    conversations = [conv for conv, m in zip(conversations, mask_succ) if m]
                    target_text = [tgt for tgt, m in zip(target_text, mask_succ) if m]

            # tokenize remaining coversations
            token_dict = self.adv_model.tokenize(conversations, target_text)

            with self.activ_extractor.capture():

                # compute per-sample activations
                with torch.inference_mode():
                    self.adv_model.set_embeddings(sample_embed)
                    self.adv_model.forward(token_dict["input_ids"], token_dict["attention_mask"], token_dict["adv_mask"])
                    sample_activs = self.activ_extractor.get_activations()

                # compute universal activations
                self.adv_model.set_embeddings(self.univ_embeds)
                self.adv_model.forward(token_dict["input_ids"], token_dict["attention_mask"], token_dict["adv_mask"])
                univ_activs = self.activ_extractor.get_activations()

            # align target mask and activations
            target_mask = token_dict["target_mask"][:, 1:]
            sample_activs = {k: v[:, :-1] for k, v in sample_activs.items()}
            univ_activs = {k: v[:, :-1] for k, v in univ_activs.items()}

            # compute loss
            criterion = ActivationLoss(loss_fn=partial(cosine_similarity_loss, target_mask=target_mask))
            loss = criterion.forward(univ_activs, sample_activs)

        # grad step
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return loss.item()
