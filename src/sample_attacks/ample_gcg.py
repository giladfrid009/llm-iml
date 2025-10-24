from src.sample_attacks.base import SampleAttack, SampleOutput
from src.adv_model import AdvModel
from src.aliases import Conv
from src.utils.logging import create_logger

from gserve.vllm_service import VLLMService
from gserve.configs import LLMConfig, ServeConfig
from vllm import SamplingParams
import torch
import msgspec
import copy


logger = create_logger(__name__)


AMPLE_MODELS = [
    "osunlp/AmpleGCG-llama2-sourced-llama2-7b-chat",  # targets llama2-7b
    "osunlp/AmpleGCG-plus-llama2-sourced-llama2-7b-chat",  # targets llama2-7b
    "osunlp/AmpleGCG-llama2-sourced-vicuna-7b",  # targets vicuna-7b v1.5
]

PROMPT_FORMAT = """### Query:{q} ### Prompt:"""


class AmpleGCG(SampleAttack):
    """
    Ample-GCG and Ample-GCG+ Attack
    [https://arxiv.org/pdf/2404.07921] and [https://arxiv.org/pdf/2410.22143]
    """

    def __init__(
        self,
        adv_model: AdvModel,
        serve_config: ServeConfig,
        model_name: str | None = None,
        memory_util: float = 0.9,
        llm_config: LLMConfig | None = None,
        sampling_params: SamplingParams | None = None,
        verbose: bool = False,
    ):
        super().__init__(adv_model, verbose)

        # try auto-selecting appropriate AmpleGCG model
        if model_name is None:
            if llm_config is not None:
                model_name = llm_config.model_name

            elif "llama-2" in adv_model.model.name_or_path.lower():
                model_name = "osunlp/AmpleGCG-plus-llama2-sourced-llama2-7b-chat"

            elif "vicuna-7b" in adv_model.model.name_or_path.lower():
                model_name = "osunlp/AmpleGCG-llama2-sourced-vicuna-7b"

            else:
                raise ValueError("Please specify a model_name for AmpleGCG attack.")

            logger.info(f"Auto-selected AmpleGCG model: {model_name}")

        if model_name not in AMPLE_MODELS:
            raise ValueError(f"Model {model_name} not in supported. Supported Models: {AMPLE_MODELS}")

        if llm_config is not None and llm_config.model_name != model_name:
            raise ValueError(
                f"LLMConfig model_name {llm_config.model_name} does not match the provided model_name {model_name}."
            )

        if llm_config is None:
            llm_config = LLMConfig(
                model_name=model_name,
                dtype="bfloat16",
                max_model_len=1024,
            )

        if sampling_params is None:
            sampling_params = SamplingParams(
                # temperature=0.0, # NOTE: if we keep at 0 then attack will become deterministic
                min_tokens=self.adv_model.num_tokens,
                max_tokens=self.adv_model.num_tokens,
            )

        llm_config.gpu_memory_utilization = memory_util

        self.model_name = model_name
        self.llm_config = llm_config
        self.serve_config = serve_config
        self.sampling_params = sampling_params

        self.model = VLLMService(self.llm_config, self.serve_config)
        self.model.start()

    def get_hparams(self) -> dict:
        return {
            "model_name": self.model_name,
            "llm_config": self.llm_config.__dict__,
            "serve_config": self.serve_config.__dict__,
            "sampling_params": msgspec.structs.asdict(self.sampling_params),
        }

    def _fmt_inputs(self, conversations: list[Conv]) -> list[str]:
        return [PROMPT_FORMAT.format(q=conv[-1]["content"]) for conv in conversations]

    def fit(
        self,
        conversations: list[Conv],
        target_texts: list[str],
        init_embeds: torch.Tensor | None = None,
    ) -> SampleOutput:
        prompts = self._fmt_inputs(conversations)
        suffixes = self.model.generate(prompts, sampling_params=self.sampling_params)

        conversations = copy.deepcopy(conversations)
        for conv, suffixes in zip(conversations, suffixes):
            conv[-1]["content"] += " " + suffixes[0]

        return SampleOutput(conversations)

    def close(self):
        self.model.shutdown()
