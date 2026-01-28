from argparse import ArgumentParser
from torch import optim
import sys
import pathlib

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from scripts.experiment import Experiment
from src.sample_attacks import SP
from src.univ_attacks import UnivAttack
from src.univ_attacks import UAP
from src.adv_model import AdvModel
from src.initialize import Initializer


class UAP_Experiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        uap_args = parser.add_argument_group("UAP Attack Parameters")

        uap_args.add_argument(
            "--num_tokens",
            type=int,
            default=20,
            metavar="NUM",
            help="Number of tokens in the adversarial trigger.",
        )

        uap_args.add_argument(
            "--skip_fooled",
            type=str,
            default="true",
            choices=["true", "false"],
            metavar="BOOL",
            help="Whether to skip samples that are already fooled by the universal trigger.",
        )

        uap_args.add_argument(
            "--skip_failed",
            type=str,
            default="true",
            choices=["true", "false"],
            metavar="BOOL",
            help="Whether to skip samples that the inner attack failed to attack.",
        )

        uap_args.add_argument(
            "--target_controls",
            type=str,
            default="false",
            choices=["true", "false"],
            metavar="BOOL",
            help="Whether control tokens are also marked as target tokens.",
        )

        inner_params = parser.add_argument_group("Inner Attack Parameters")

        inner_params.add_argument(
            "--inner_attack_lr",
            type=float,
            default=5e-3,
            metavar="FLOAT",
            help="Learning rate for the inner attack.",
        )

        inner_params.add_argument(
            "--inner_attack_steps",
            type=int,
            default=15,
            metavar="INT",
            help="Number of steps for the inner attack.",
        )

        inner_params.add_argument(
            "--inner_attack_target_matching",
            type=str,
            default="false",
            choices=["true", "false"],
            metavar="BOOL",
            help="Whether to use target matching for the inner attack.",
        )

        inner_params.add_argument(
            "--kv_caching",
            type=str,
            default="true",
            choices=["true", "false"],
            metavar="BOOL",
            help="Whether to use kv-caching for the inner attack.",
        )
        
        inner_params.add_argument(
            "--pert_path",
            type=str,
            default=None,
            metavar="PATH",
            help="Path to initial perturbation embeddings (optional).",
        )

        parser.set_defaults(
            project_name="UAP",
            model="meta-llama/Llama-2-7b-chat-hf",
            train_batch=1,
            skip_fooled="true",
            skip_failed="true",
            target_controls="false",
            # inner-attack params
            kv_caching="true",
            inner_attack_lr=5e-3,
            inner_attack_steps=15,
            inner_attack_target_matching="false",
        )

    def create_adversarial_model(self, model, tokenizer) -> AdvModel:
        adv_model = AdvModel(model, tokenizer, num_tokens=self.args().num_tokens, add_spaces=False, adv_suffix=True)
        
        if self.args().pert_path is not None:
            embeds = Initializer.load(adv_model, self.args().pert_path, batch_size=1)
        else:
            embeds = Initializer.random_normal(adv_model, std=0.1)
        
        adv_model.set_embeddings(embeds)
        return adv_model

    def initialize_attack(
        self,
        adv_model: AdvModel,
        evaluators,
        eval_metric,
        eval_freq,
        mixed_precision,
        gen_config,
        metric_tracker,
    ) -> UnivAttack:
        args = self.args()
        
        inner_attack = SP(
            adv_model,
            optim_factory=lambda params: optim.AdamW(params, lr=args.inner_attack_lr),
            steps=args.inner_attack_steps,
            target_matching=args.inner_attack_target_matching == "true",
            kv_caching=args.kv_caching == "true",
        )

        return UAP(
            adv_model=adv_model,
            inner_attack=inner_attack,
            evaluators=evaluators,
            eval_metric=eval_metric,
            eval_freq=eval_freq,
            gen_config=gen_config,
            mixed_precision=mixed_precision,
            metric_tracker=metric_tracker,
            # specialized args
            skip_already_fooled=args.skip_fooled == "true",
            skip_failed_attacks=args.skip_failed == "true",
        )


if __name__ == "__main__":
    UAP_Experiment().main()
