import torch
import random
import argparse
import sys
import time

from tqdm.auto import tqdm

from src.utils import env
from src.utils.logging import create_logger, setup_logging, loglevel_names
from src.data import TableLoader
from src.adv_model import AdvModel
from src.config import GenConfig
from src.sample_attacks import SP
from src.initialize import Initializer
from src.fgsm_optim import FGSM
from src.utils.trackers import MetricTracker

from scripts.utils.load_model import SUPPORTED_MODELS, load_model
from scripts.utils.load_dataset import SUPPORTED_DATASETS, load_dataset
from scripts.utils.load_evaluator import SUPPORTED_EVALUATORS, load_evaluators
from typing import Callable, Iterable
import copy

from src.sample_attacks.base import SampleOutput
from src.aliases import Conv
import math

logger = create_logger(__name__)


class SP_Sample(SP):
    """
    Soft Prompt Threats Attack: optimizes continuous adversarial embeddings
    [https://arxiv.org/pdf/2402.09063]
    """

    def __init__(
        self,
        adv_model: AdvModel,
        optim_factory: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer],
        steps: int = 100,
        target_matching: bool = False,
        target_loss: float | None = None,
        noise_scale: float = 0.0,
        *,
        kv_caching: bool = True,
        mixed_precision: bool = False,
        verbose: bool = True,
    ):
        """
        Args:
            adv_model (AdvModel): The adversarial model to attack.
            optim_factory (Callable[Iterable[torch.Tensor], torch.optim.Optimizer]):
                A factory function that creates an optimizer given the parameters to optimize.
            steps (int): Number of optimization steps.
            target_matching (bool): Whether to stop optimizing a sample once it achieves perfect target matching.
            target_loss (float | None): If specified, stop optimizing a sample once its loss is below this threshold.
            noise_scale (float): Standard deviation of Gaussian noise added to the initial embeddings.
            kv_caching (bool): Whether to use kv-caching for the constant part of the input.
            mixed_precision (bool): Whether to use mixed precision training.
            verbose (bool): Whether to display a progress bar.
        """
        super().__init__(
            adv_model=adv_model,
            optim_factory=optim_factory,
            steps=steps,
            target_matching=target_matching,
            target_loss=target_loss,
            noise_scale=noise_scale,
            kv_caching=kv_caching,
            mixed_precision=mixed_precision,
            verbose=verbose,
        )

    def _create_embeddings(
        self,
        num_inputs: int,
        init_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if init_embeds is not None:
            init_embeds = init_embeds  # NOTE: WE DO NOT CLONE
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
        encodings = self.adv_model.tokenize(conversations, target_texts)

        # compute kv-cache
        if self.kv_caching:
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
                step_enc = encodings.copy()
                step_enc["kv_cache"] = copy.deepcopy(encodings.kv_cache) if "kv_cache" in step_enc else None

                # select only unfinished samples if early stopping is enabled
                if self.target_matching or self.target_loss is not None:
                    step_enc = self._masked_select(step_enc, ~finished)
                    optim_embeds = adv_embeds[~finished]

                with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                    # forward pass
                    result = self.adv_model.forward(
                        input_ids=step_enc.input_ids,
                        attention_mask=step_enc.attention_mask,
                        adv_mask=step_enc.adv_mask,
                        past_key_values=step_enc.kv_cache,
                        adv_embeds=optim_embeds,
                    )

                    # update early stopping based on matching
                    if self.target_matching:
                        status = self._check_matching(result.logits, step_enc.input_ids, step_enc.target_mask)
                        finished[~finished] = status
                        if finished.all():
                            break

                    sample_loss = self.sample_criterion(
                        logits=result.logits,
                        input_ids=step_enc.input_ids,
                        target_mask=step_enc.target_mask,
                    )

                    # update early stopping based on loss
                    if self.target_loss is not None:
                        status = sample_loss <= self.target_loss
                        finished[~finished] |= status
                        if finished.all():
                            break

                        sample_loss = sample_loss[~status]

                    loss = sample_loss.sum()

                # backward pass and optimization step
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                # update progress bar
                num_remaining = (~finished).sum().item()
                remaining = f"{num_remaining}/{len(conversations)}"
                pbar.set_postfix(loss=loss.item(), remaining=remaining)

                LOGS["loss"].append(loss.item() / result.logits.size(0))
                LOGS["remaining"].append(num_remaining)

            # close pbar
            pbar.n = pbar.total
            pbar.close()

        return SampleOutput(conversations, adv_embeds, logs=LOGS)


class SoftIndivRunner:
    def __init__(self) -> None:
        self._parsed_args = None

    def args(self) -> argparse.Namespace:
        if self._parsed_args is None:
            raise ValueError("Arguments have not been parsed yet. Call _parse_args() first.")
        return self._parsed_args

    def _parse_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument(
            "--model",
            type=str,
            choices=SUPPORTED_MODELS,
            default="meta-llama/Llama-2-7b-chat-hf",
            metavar="MODEL",
            help=f"The model name to attack. Available models: {SUPPORTED_MODELS}",
        )

        parser.add_argument(
            "--dataset",
            type=str,
            choices=SUPPORTED_DATASETS,
            default="advbench",
            metavar="DATASET",
            help=f"The datasets to use. Available datasets: {SUPPORTED_DATASETS}",
        )

        parser.add_argument(
            "--evaluator",
            type=str,
            nargs="+",
            choices=SUPPORTED_EVALUATORS,
            default=["strong-reject", "keyword-matching"],
            metavar="EVALUATOR",
            help=f"The attack evaluators to use. Available evaluators: {SUPPORTED_EVALUATORS}",
        )

        parser.add_argument(
            "--run_name",
            type=str,
            default=time.strftime("%Y-%m-%d_%H-%M-%S"),
            metavar="NAME",
            help="The name of the run, used for logging.",
        )

        parser.add_argument(
            "--train_batch",
            type=int,
            default=50,
            metavar="SIZE",
            help="The training batch size.",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=random.randint(0, 1000000),
            help="Random seed for reproducibility.",
        )

        parser.add_argument(
            "--log_level",
            type=str,
            choices=loglevel_names(),
            default="INFO",
            metavar="LEVEL",
            help=f"Logging level to python-logger. Available levels: {loglevel_names()}",
        )

        attack_args = parser.add_argument_group("Base attack parameters")

        attack_args.add_argument(
            "--eval_metric",
            type=str,
            default=None,
            metavar="NAME",
            help="The evaluation metric to use for selecting the best adversarial prompt. "
            "If not specified, the default metric of the first evaluator will be used.",
        )

        attack_args.add_argument(
            "--eval_freq",
            type=int,
            default=10,
            metavar="NUM",
            help="Frequency of evaluation during training, in steps.",
        )

        attack_args.add_argument(
            "--use_amp",
            choices=["true", "false"],
            metavar="BOOL",
            default="false",
            help="Whether to use automatic mixed precision (AMP) for training.",
        )

        attack_args.add_argument(
            "--num_tokens",
            type=int,
            default=20,
            metavar="NUM",
            help="Number of tokens in the adversarial trigger.",
        )

        attack_args.add_argument(
            "--lr",
            type=float,
            metavar="FLOAT",
            default=0.001,
            help="Learning rate for FGSM optimizer.",
        )

        gen_args = parser.add_argument_group("Generation parameters")

        gen_args.add_argument(
            "--do_sample",
            choices=["true", "false"],
            metavar="BOOL",
            default="true",
            help="Whether to use sampling for generation.",
        )

        gen_args.add_argument(
            "--temperature",
            type=float,
            default=None,
            metavar="TEMP",
            help="Sampling temperature for generation.",
        )

        gen_args.add_argument(
            "--top_p",
            type=float,
            default=None,
            metavar="P",
            help="Nucleus sampling top-p value for generation.",
        )

        gen_args.add_argument(
            "--top_k",
            type=int,
            default=None,
            metavar="K",
            help="Top-k sampling value for generation.",
        )

        gen_args.add_argument(
            "--max_new_tokens",
            type=int,
            default=512,
            metavar="NUM",
            help="Maximum number of new tokens to generate.",
        )

        stop_args = parser.add_argument_group("Stopping Criteria")

        stop_args.add_argument(
            "--num_iters",
            type=int,
            default=500,
            metavar="NUM",
            help="The number of overall training iterations.",
        )

        stop_args.add_argument(
            "--patience",
            type=int,
            default=15,
            metavar="NUM",
            help="The number of evaluation steps with no improvement to wait before stopping.",
        )

        attack_args.add_argument(
            "--attack_train",
            action="store_true",
            help="Whether to attack the training set.",
        )

        attack_args.add_argument(
            "--attack_val",
            action="store_true",
            help="Whether to attack the validation set.",
        )

        self._parsed_args = parser.parse_args()
        args = self.args()

        # print the parsed arguments
        print()
        print("Parsed arguments:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
        print()

        return args

    def prepare_environment(self, seed: int | None):
        if seed is None:
            seed = random.randint(0, 10000)
        logger.info(f"Random seed: {seed}")

        torch.set_float32_matmul_precision("high")
        env.prepare_environment()
        env.set_seed(seed)

    def run(self):
        args = self.args()

        if not torch.cuda.is_available():
            logger.error("No GPU available. Exiting.")
            sys.exit(1)

        with MetricTracker.create(
            self.args().run_name,
            kind="wandb",
            root_dir=f"logs/{args.model.split('/')[-1]}/{args.dataset}",
            project="LLM-IML",
        ) as metric_tracker:
            logger.info(f"Loading dataset: {args.dataset}")
            ds_train, ds_val, ds_test = load_dataset(args.dataset)
            dl_train = TableLoader(ds_train, batch_size=args.train_batch, shuffle=False)
            dl_val = TableLoader(ds_val, batch_size=args.train_batch, shuffle=False)
            dl_test = TableLoader(ds_test, batch_size=args.train_batch, shuffle=False)

            logger.info(f"Loaded datasets with sample counts: (train, val, test) = ({len(ds_train)}, {len(ds_val)}, {len(ds_test)}).")

            logger.info(f"Loading evaluator: {args.evaluator}")
            device_count = torch.cuda.device_count()
            gpus = [] if device_count <= 1 else list(range(device_count))[1:]
            logger.info(f"GPUs available for evaluators: {gpus}")
            evaluators = load_evaluators(args.evaluator, gpus=gpus)

            logger.info(f"Loading model: {args.model}")
            model, tokenizer = load_model(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")

            adv_model = AdvModel(model, tokenizer, num_tokens=self.args().num_tokens, add_spaces=False, adv_suffix=True)

            gen_config = GenConfig(
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample.lower() == "true",
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )

            judge_evaluator = evaluators[0]
            eval_metric_name: str = args.eval_metric or judge_evaluator.default_metric
            logger.info(f"Using evaluation metric: {eval_metric_name}")

            metric_tracker.set_tags(
                model=args.model,
                num_tokens=adv_model.num_tokens,
                attack="SP_Sample",
                dataset=args.dataset,
                evaluators=", ".join(args.evaluator),
            )

            total_succ = 0
            total_tot = 0

            if self.args().attack_train:
                train_succ, train_tot = self.attack_dataset(
                    dl_train,
                    "train",
                    metric_tracker,
                    adv_model,
                    gen_config,
                    judge_evaluator,
                    eval_metric_name,
                )
                total_succ += train_succ
                total_tot += train_tot

            if self.args().attack_val:
                val_succ, val_tot = self.attack_dataset(
                    dl_val,
                    "val",
                    metric_tracker,
                    adv_model,
                    gen_config,
                    judge_evaluator,
                    eval_metric_name,
                )
                total_succ += val_succ
                total_tot += val_tot

            test_succ, test_tot = self.attack_dataset(
                dl_test,
                "test",
                metric_tracker,
                adv_model,
                gen_config,
                judge_evaluator,
                eval_metric_name,
            )
            total_succ += test_succ
            total_tot += test_tot

            metric_tracker.report_globals(
                {
                    "final_asr": total_succ / total_tot if total_tot > 0 else 0,
                    "num_success": total_succ,
                    "num_total": total_tot,
                }
            )

            logger.info(
                f"Final Attack Success Rate (ASR): {total_succ}/{total_tot} = {total_succ / total_tot:.4f}"
                if total_tot > 0
                else "No attacks performed."
            )

            for ev in evaluators:
                ev.close()

    def attack_dataset(self, dl, split_name, metric_tracker, adv_model, gen_config, judge_evaluator, eval_metric_name):
        args = self.args()
        num_success = 0
        num_total = 0

        # Initialize response column in the dataset
        dl.df["response"] = None
        successful_responses = []
        global_index = 0

        with tqdm(dl, desc=f"Batches ({split_name})", leave=False) as batch_pbar:
            for batch_num, data in enumerate(batch_pbar):
                input_texts, target_text = data["prompt"], data["target"]
                conversations = [[{"role": "user", "content": prm}] for prm in input_texts]

                # create batch embeds and optimizer
                embeds = Initializer.random_normal(adv_model, std=0.1, batch_size=len(input_texts))
                optim = FGSM([embeds], lr=args.lr)

                best_metric = torch.zeros(len(input_texts), device=adv_model.device, dtype=torch.bool)

                previous_best_sum = 0
                patience_counter = 0

                attack = SP_Sample(
                    adv_model=adv_model,
                    optim_factory=lambda _: optim,
                    steps=args.eval_freq,
                    target_matching=False,
                    target_loss=None,
                    noise_scale=0,
                    kv_caching=False,
                    mixed_precision=args.use_amp.lower() == "true",
                    verbose=True,
                )

                iters = math.ceil(args.num_iters / args.eval_freq)
                with tqdm(range(iters), leave=False, desc="Overall iters") as step_pbar:
                    for _ in step_pbar:
                        sample_result = attack.fit(conversations, target_texts=target_text, init_embeds=embeds)

                        with torch.inference_mode():
                            sample_responses = adv_model.chat(
                                conversations=sample_result.conversations,
                                adv_embeds=sample_result.adv_embeds,
                                config=gen_config,
                            )

                        eval_result = judge_evaluator.eval_batch(input_texts, sample_responses)
                        eval_metric = torch.tensor(eval_result[eval_metric_name], device=adv_model.device)
                        is_successful = eval_metric >= 1.0

                        previous_best_metric = best_metric.clone()
                        best_metric = best_metric | is_successful

                        current_sum = best_metric.sum().item()
                        if current_sum > previous_best_sum:
                            previous_best_sum = current_sum
                            patience_counter = 0

                            # Collect responses for newly successful samples
                            new_improved = is_successful & ~previous_best_metric
                            for i in range(len(input_texts)):
                                if new_improved[i]:
                                    dl.df.loc[global_index + i, "response"] = sample_responses[i]
                                    successful_responses.append(sample_responses[i])
                        else:
                            patience_counter += 1

                        if patience_counter >= args.patience:
                            break

                        if best_metric.all():
                            break

                        step_pbar.set_postfix(
                            running_asr=f"{best_metric.sum().item() / len(input_texts):.4f}",
                            patience=patience_counter,
                        )

                global_index += len(input_texts)
                num_success += best_metric.sum().item()
                num_total += len(input_texts)

                batch_pbar.set_postfix(
                    asr=f"{num_success / num_total:.4f}",
                    batch_asr=f"{best_metric.sum().item() / len(input_texts):.4f}",
                )

        if log_dir := metric_tracker.log_dir:
            dl.df.to_csv(f"{log_dir}/{split_name}_results.csv", index=False)
            logger.info(f"Saved {split_name} results to '{log_dir}/'.")

        return num_success, num_total

    def main(self):
        try:
            self._parse_args()
            setup_logging(level=self.args().log_level)
            self.prepare_environment(seed=self.args().seed)
            self.run()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
            sys.exit(0)


if __name__ == "__main__":
    SoftIndivRunner().main()
