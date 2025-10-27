from __future__ import annotations
import torch
from torch import nn
from typing import Iterator
import copy

from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from src.aliases import Conv
from src import tokenize
from src.utils.torch import extract_device
from src.config import GenConfig
from src.discretize import Discretize
from src.utils.logging import create_logger


logger = create_logger(__name__)


class AdverEmbedding(nn.Module):
    def __init__(self, embedder: torch.nn.Embedding):
        super().__init__()
        self.embedder = embedder
        self.device = extract_device(embedder)

        # internal params for caching
        self._embed_dim: int = None  # type: ignore
        self._embed_dtype: torch.dtype = None  # type: ignore

    @property
    def embed_dim(self) -> int:
        if self._embed_dim is not None:
            return self._embed_dim
        test_input = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        dim = self.embedder(test_input).size(-1)
        self._embed_dim = dim
        return dim

    @property
    def embed_dtype(self) -> torch.dtype:
        if self._embed_dtype is not None:
            return self._embed_dtype
        test_input = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        dtype = self.embedder(test_input).dtype
        self._embed_dtype = dtype
        return dtype

    def forward(
        self,
        inputs: torch.Tensor,
        adv_embeds: torch.Tensor | None = None,
        adv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Input token IDs of shape (batch_size, sequence_length).
            adv_embeds (torch.Tensor | None): Adversarial embeddings of shape (batch_size, num_adv_tokens, hidden_size).
                If batch_size is 1, it will be broadcasted to match input batch size.
            adv_mask (torch.Tensor | None): Boolean mask of shape (batch_size, sequence_length) indicating positions of adversarial tokens.

        Returns:
            torch.Tensor: Embedded inputs of shape (batch_size, sequence_length, hidden_size).
        """
        if not inputs.ndim == 2:
            raise ValueError("`inputs` must be a 2D tensor (batch_size, sequence_length)")

        if (adv_embeds is None) != (adv_mask is None):
            raise ValueError("Both adv_embeds and adv_mask should be None or not None")

        if adv_mask is None or adv_embeds is None:
            return self.embedder.forward(inputs)

        B, N, H = adv_embeds.shape
        B, S = inputs.shape

        if adv_embeds.size(0) == 1:
            # project to batch size
            adv_embeds = adv_embeds.expand(inputs.size(0), *adv_embeds.shape[1:])

        # verify shapes
        assert inputs.shape == (B, S), "Inputs shape mismatch"
        assert adv_mask.shape == (B, S), "Adversarial mask shape mismatch"
        assert adv_embeds.shape == (B, N, H), "Adversarial embeddings shape mismatch"

        # verify that number of True in adv_mask matches adv_embeds size(1)
        if not torch.all(adv_mask.sum(dim=1) == N):
            raise RuntimeError(f"Number of adversarial tokens in adv_mask does not match num_adv_tokens ({N})")

        # Zero out adversarial token ids before embedding and then patch with adversarial embeddings
        clean_inputs = inputs.masked_fill(adv_mask, 0)
        embedded = self.embedder.forward(clean_inputs)
        return embedded.masked_scatter(mask=adv_mask.unsqueeze(-1), source=adv_embeds)


class AdvModel(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        num_tokens: int = 10,
        *,
        adv_token: str = "[ADV]",
        add_spaces: bool = False,
        adv_suffix: bool = True,
    ):
        """
        An adversarially-augmented language model.
        It wraps around a pre-trained language model and tokenizer, allowing for the injection of adversarial tokens
        into input conversations and the use of adversarial embeddings during model inference and generation.

        Args:
            model (PreTrainedModel): A pre-trained language model from the Hugging Face Transformers library.
            tokenizer (PreTrainedTokenizer): A tokenizer corresponding to the pre-trained model.
            num_tokens (int): Number of adversarial tokens to be injected into the input.
            adv_token (str): The special token used to represent adversarial tokens in the input text.
            add_spaces (bool): If True, separates the adversarial tokens with spaces when injecting into the text.
            adv_suffix (bool): Whether to add the adversarial tokens as a suffix (True) or prefix (False) to the last message in the conversation.
        """
        super().__init__()

        # model
        self.model = model.eval()
        self.model.requires_grad_(False)

        # tokenizer
        self.tokenizer = tokenizer
        self.adv_token = adv_token
        if self.adv_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [self.adv_token]})  # type: ignore

        # adv embedder
        self.orig_embedder: torch.nn.Embedding = self.model.get_input_embeddings()  # type: ignore
        self.adv_embedder = AdverEmbedding(self.orig_embedder)

        # params
        self.device = extract_device(model)
        self.num_tokens = num_tokens
        self.add_spaces = add_spaces
        self.adv_suffix = adv_suffix

        # adversarial embeddings
        self.adv_embeds: nn.Parameter | None = None

    def get_hparams(self) -> dict:
        return {
            "model_name": self.model.name_or_path,
            "tokenizer_name": self.tokenizer.name_or_path,
            "adv_token": self.adv_token,
            "num_tokens": self.num_tokens,
            "embed_dim": self.adv_embedder.embed_dim,
            "add_spaces": self.add_spaces,
            "adv_suffix": self.adv_suffix,
        }

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """
        Yields the trainable parameters of the module.
        Only the adversarial embeddings are trainable.

        Yields:
            Iterator[nn.Parameter]: Trainable parameters.
        """
        if self.adv_embeds is None:
            raise ValueError("Adversarial embeddings are not set. Please set them using `set_embeddings` method.")

        yield self.adv_embeds

    @torch.no_grad()
    def discretize(self) -> None:
        if self.adv_embeds is None:
            raise ValueError("Adversarial embeddings are not set. Please set them using `set_embeddings` method.")

        closest_indices = Discretize.cosine_similarity(self.adv_embeds, self.orig_embedder.weight)
        discrete_embeds = self.embed(closest_indices)
        self.set_embeddings(discrete_embeds, strict=True)

    def train(self, mode: bool = True) -> AdvModel:
        """
        Overrides the default train() method to ensure the inner model remains in evaluation mode.

        Args:
            mode (bool): Training mode flag (True for training, False for evaluation).

        Returns:
            AdvModel: Self.
        """
        self.training = mode
        return self

    def set_embeddings(self, adv_embeds: torch.Tensor, strict: bool = True) -> None:
        """
        Set the adversarial embeddings, and the corresponding number of attacked tokens.
        If batch_size is 1, the embeddings will be broadcasted to input batch size.

        Args:
            adv_embeds (torch.Tensor): Adversarial embeddings of shape (batch_size, num_tokens, embed_dim).
            strict (bool): If True, number of adversarial tokens must match to `self.num_tokens`.
        """
        assert adv_embeds.ndim == 3, "Adversarial embeddings must be a 3D tensor (batch_size, num_tokens, embed_dim)"
        assert adv_embeds.size(2) == self.adv_embedder.embed_dim, (
            "Adversarial embeddings must match the embed_dim of the model"
        )

        num_tokens = adv_embeds.size(1)
        if num_tokens != self.num_tokens and strict:
            raise ValueError(f"Number of adversarial tokens must be {self.num_tokens}, but got {num_tokens}.")

        if torch.is_inference(adv_embeds):
            with torch.enable_grad():
                logger.warning("Adversarial embeddings are inference tensor. Cloning with grad enabled.")
                adv_embeds = adv_embeds.clone().detach()

        self.adv_embeds = nn.Parameter(adv_embeds)
        self.num_tokens = num_tokens

    def get_embeddings(self, clone: bool = False) -> torch.Tensor:
        """
        Get the adversarial embeddings tensor.
        It is highly recommended to get the embedding through this method.

        Args:
            clone (bool): If True, returns a cloned and detached tensor, otherwise returns the original tensor.

        Returns:
            torch.Tensor: Adversarial embeddings of shape (batch_size, num_tokens, embed_dim).
        """
        if self.adv_embeds is None:
            raise ValueError("Adversarial embeddings are not set. Please set them using `set_embeddings` method.")
        return self.adv_embeds.clone().detach() if clone else self.adv_embeds

    def inject_tokens(
        self,
        conversations: list[Conv],
        num_tokens: int | None = None,
        add_spaces: bool | None = None,
        adv_suffix: bool | None = None,
    ) -> list[Conv]:
        """
        Injects adversarial tokens to the last message in each conversation.
        A clone of the input conversations is returned.

        Note: if `None` is passed to any of the optional arguments, the corresponding
        default attribute of `self` will be used.

        Args:
            messages (list[Conv]): A batch of conversations.
            num_tokens (int | None): Number of adversarial tokens to inject.
            add_spaces (bool | None): If True, separates the adversarial tokens with spaces.
            adv_suffix (bool | None): Whether to add the adversarial tokens as a suffix or prefix to the last message.

        Returns:
            (list[Conv]): A batch of conversations with adversarial tokens injected into the last message.
        """
        if num_tokens is None:
            num_tokens = self.num_tokens

        if add_spaces is None:
            add_spaces = self.add_spaces

        if adv_suffix is None:
            adv_suffix = self.adv_suffix

        conversations = copy.deepcopy(conversations)
        separator = " " if add_spaces else ""
        adv_block = separator.join([self.adv_token] * num_tokens)

        for conv in conversations:
            # don't inject if adv_token already present
            if any(self.adv_token in msg["content"] for msg in conv):
                continue

            last_msg = conv[-1]
            content = last_msg["content"]

            if adv_suffix:
                spacer = " " if add_spaces and not content.endswith(" ") else ""
                last_msg["content"] = f"{content}{spacer}{adv_block}"
            else:
                spacer = " " if add_spaces and not content.startswith(" ") else ""
                last_msg["content"] = f"{adv_block}{spacer}{content}"

        return conversations

    def repl_tokens(
        self,
        conversations: list[Conv],
        repl_ids: list[list[int]],
    ) -> list[Conv]:
        """
        Replace all occurrences of the adversarial token in the conversations with the provided token IDs.

        Args:
            conversations (list[Conv]): A batch of conversations, where each conversation is a list of messages.
            repl_ids (list[list[int]]): A list of lists of token IDs to replace the adversarial tokens with.

        Returns:
            list[Conv]: The modified conversations with adversarial tokens replaced by the specified token IDs.
        """
        return tokenize.replace_tokens(
            tokenizer=self.tokenizer,
            conversations=conversations,
            repl_ids=repl_ids,
            adv_token=self.adv_token,
        )

    def tokenize(
        self,
        conversations: list[Conv],
        target_texts: list[str] | None = None,
    ) -> BatchEncoding:
        """
        Tokenize the input and target texts.

        Args:
            conversations (list[Conv]): A batch of conversations, where each conversation is a list of messages.
            target_texts (list[str] | None): List of target texts. If None, only input texts are tokenized.

        Returns:
            BatchEncoding:
            An batch encoding object containing the tokenized data with the following keys:
                - `input_ids` (torch.IntTensor): Token IDs of the entire tokenized texts.
                - `attention_mask` (torch.BoolTensor): Attention mask of the entire tokenized texts.
                - `adv_mask` (torch.BoolTensor): Mask for the adversarial tokens.
                - `const_idx` (torch.IntTensor): First index of the adversarial tokens in each conversation, used for KV-caching.
                - `target_mask` (torch.BoolTensor): Only if `target_texts != None`. Mask for the target tokens
        """
        if target_texts is not None:
            tokenized = tokenize.chat_with_targets(
                tokenizer=self.tokenizer,
                conversations=conversations,
                target_texts=target_texts,
                adv_token=self.adv_token,
            )
        else:
            tokenized = tokenize.chat(
                tokenizer=self.tokenizer,
                conversations=conversations,
                adv_token=self.adv_token,
            )

        return tokenized.to(self.device)

    def embed(
        self,
        inputs: torch.Tensor,
        adv_embeds: torch.Tensor | None = None,
        adv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.adv_embedder.forward(inputs, adv_embeds, adv_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        adv_mask: torch.Tensor | None,
        adv_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        """
        Performs forward pass of the model with optional adversarial embeddings.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input tokens.
            adv_mask (torch.Tensor | None): Mask indicating positions of adversarial tokens.
            adv_embeds (torch.Tensor | None): Adversarial embeddings to be used during the forward pass.
                If None, uses the default adversarial embeddings set in the model (if necessary).
            **kwargs: Additional keyword arguments to be passed to the model's forward method.
        """

        # use default adv_embeds if they should be used but not provided
        if adv_embeds is None and adv_mask is not None and torch.any(adv_mask):
            adv_embeds = self.adv_embeds

        if adv_mask is not None and adv_embeds is not None:
            inputs_embeds = self.embed(input_ids, adv_embeds, adv_mask)
        else:
            inputs_embeds = self.embed(input_ids, None, None)

        return self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        adv_mask: torch.Tensor | None,
        adv_embeds: torch.Tensor | None,
        config: GenConfig | None = None,
        **kwargs,
    ) -> GenerateDecoderOnlyOutput | torch.Tensor:
        """
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input tokens.
            adv_mask (torch.Tensor | None): Mask indicating positions of adversarial tokens.
            adv_embeds (torch.Tensor | None): Adversarial embeddings to be used during generation.
                If None, uses the default adversarial embeddings set in the model (if necessary).
            config (GenConfig | None): Generation configuration. If None, uses the default generation configuration.
            **kwargs: Additional keyword arguments to be passed to the model's generate method.

        Returns:
            (GenerateDecoderOnlyOutput | torch.Tensor): The generated sequences or a generation output object.
        """
        if config is None:
            config = GenConfig()

        # use default adv_embeds if they should be used but not provided
        if adv_embeds is None and adv_mask is not None and torch.any(adv_mask):
            adv_embeds = self.adv_embeds

        if generation_config := copy.deepcopy(self.model.generation_config):
            config.patch_other(generation_config)

        if adv_mask is not None and adv_embeds is not None:
            inputs_embeds = self.embed(input_ids, adv_embeds, adv_mask)
        else:
            inputs_embeds = self.embed(input_ids, None, None)

        return self.model.generate(
            inputs=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_model_defaults=False,
            **kwargs,
        )  # type: ignore

    @torch.inference_mode()
    def chat(
        self,
        conversations: list[Conv],
        adv_embeds: torch.Tensor | None = None,
        config: GenConfig | None = None,
        **kwargs,
    ) -> list[str]:
        """
        Generates adversarial texts for a batch of conversations.

        Args:
            conversations (list[Conv]): A batch of conversations, where each conversation is a list of messages.
            adv_embeds (torch.Tensor | None): The adversarial embeddings to be used during generation.
                If None, uses the default adversarial embeddings set in the model (if necessary).
            config (GenConfig | None): Generation configuration. If None, uses the default generation configuration.

        Returns:
            list[str]: List of generated adversarial texts.
        """
        encodings = self.tokenize(conversations)

        result: torch.Tensor = self.generate(
            input_ids=encodings.input_ids,
            attention_mask=encodings.attention_mask,
            adv_mask=encodings.adv_mask,
            adv_embeds=adv_embeds,
            config=config,
            return_dict_in_generate=False,
            **kwargs,
        )  # type: ignore

        return self.tokenizer.batch_decode(result, skip_special_tokens=True)
