import torch
from torch import nn
from typing import Iterator
import copy

from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from src import tokenize
from src import utils


class AdverEmbedding(nn.Module):
    def __init__(self, embedder: nn.Module):
        super().__init__()
        self.embedder = embedder
        self.device = utils.extract_device(embedder)

        # internal params for caching
        self._embed_dim: int = None
        self._embed_dtype: torch.dtype = None

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

        if (adv_embeds is None) != (adv_mask is None):
            raise ValueError("Both adv_embeds and adv_mask should be None or not None")

        if adv_mask is None:
            return self.embedder(inputs)

        if inputs.shape != adv_mask.shape:
            raise ValueError(f"Shape mismatch: input {inputs.shape}, adv_mask {adv_mask.shape}")

        assert inputs.ndim + 1 == adv_embeds.ndim
        assert inputs.size(0) == adv_embeds.size(0)
        assert inputs.ndim == adv_mask.ndim

        embedded_clean = self.embedder(inputs[~adv_mask])
        embedded = torch.zeros(*inputs.shape, self.embed_dim, dtype=self.embed_dtype, device=self.device)
        embedded = embedded.masked_scatter(mask=~adv_mask.unsqueeze(-1), source=embedded_clean)
        return embedded.masked_scatter(mask=adv_mask.unsqueeze(-1), source=adv_embeds)


class AdverModel(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        num_tokens: int = 10,
        adv_token: str = "[ADV]",
    ):
        super().__init__()

        # model
        self.model = model.eval()
        self.model.requires_grad_(False)

        # tokenizer
        self.tokenizer = tokenizer
        self.adv_token = adv_token
        if self.adv_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [self.adv_token]})

        # adv embedder
        self.orig_embedder = self.model.get_input_embeddings()
        self.adv_embedder = AdverEmbedding(self.orig_embedder)
        self.model.set_input_embeddings(self.adv_embedder)

        # params
        self.device = utils.extract_device(model)
        self.num_tokens = num_tokens

        # adversarial embeddings
        self.adv_embeds: torch.Tensor | None = None

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """
        Yields the trainable parameters of the module.
        Only the adversarial embeddings are trainable.

        Yields:
            Iterator[nn.Parameter]: Trainable parameters.
        """
        yield self.adv_embeds

    def train(self, mode: bool = True) -> "AdverModel":
        """
        Overrides the default train() method to ensure the inner model remains in evaluation mode.

        Args:
            mode (bool): Training mode flag (True for training, False for evaluation).

        Returns:
            AdverModel: Self.
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
        assert adv_embeds.size(2) == self.adv_embedder.embed_dim, "Adversarial embeddings must match the embed_dim of the model"

        num_tokens = adv_embeds.size(1)
        if num_tokens != self.num_tokens and strict:
            raise ValueError(f"Number of adversarial tokens must be {self.num_tokens}, but got {num_tokens}.")

        self.adv_embeds = adv_embeds
        self.num_tokens = num_tokens

    def get_embeddings(self, clone: bool = False) -> torch.Tensor:
        """
        Get the adversarial embeddings tensor.
        It is highly recommended to get the embedding throught this method.

        Args:
            clone (bool): If True, returns a cloned and detached tensor, otherwise returns the original tensor.

        Returns:
            torch.Tensor: Adversarial embeddings of shape (batch_size, num_tokens, embed_dim).
        """
        if self.adv_embeds is None:
            raise ValueError("Adversarial embeddings are not set. Please set them using `set_embeddings` method.")
        return self.adv_embeds.clone().detach() if clone else self.adv_embeds

    def inject_tokens(self, conversations: list[list[dict[str, str]]]) -> list[list[dict[str, str]]]:
        """
        Injects adversarial tokens to the last message in each conversation.
        Note that the injected tokens do not change the original conversations.

        Args:
            messages (list[list[dict[str, str]]]): A batch of conversations. Each conversation is a list of messages,
                where each message is a dictionary with keys "role" and "content".

        Returns:
            list[list[dict[str, str]]]: A batch of conversations with adversarial tokens injected into the last message.
        """
        conversations = copy.deepcopy(conversations)
        for conv in conversations:
            last_msg = conv[-1]
            last_msg["content"] += self.adv_token * self.num_tokens
        return conversations

    def tokenize(
        self,
        conversations: list[list[dict[str, str]]],
        target_texts: list[str] | None = None,
    ) -> BatchEncoding:
        """
        Tokenize the input and target texts.

        Args:
            conversations (list[list[dict[str, str]]]): A batch of conversations, where each conversation is a list of messages.
                Each message is a dictionary with keys "role" and "content".
            target_texts (list[str] | None): List of target texts. If None, only input texts are tokenized.

        Returns:
            (BatchEncoding):
            An instance of `BatchEncoding` containing the tokenized data with the following keys:
                - `input_ids` (torch.IntTensor): Token IDs of the entire tokenized texts.
                - `attention_mask` (torch.BoolTensor): Attention mask of the entire tokenized texts.
                - `adv_mask` (torch.BoolTensor): Mask for the adversarial tokens.
                - `const_idx` (torch.IntTensor): Only if `target_texts != None`. Index values for the constant tokens for KV-cache.
                - `target_mask` (torch.BoolTensor): Only if `target_texts != None`. Mask for the target tokens
        """

        # add adver tokens
        conversations = self.inject_tokens(conversations)

        if target_texts is not None:
            tokenized = tokenize.chat_with_targets(
                tokenizer=self.tokenizer,
                conversations=conversations,
                target_texts=target_texts,
                adver_token=self.adv_token,
            )
        else:
            tokenized = tokenize.chat(
                tokenizer=self.tokenizer,
                conversations=conversations,
                adver_token=self.adv_token,
            )

        return tokenized.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        adv_mask: torch.Tensor | None = None,
        **kwargs,
    ):

        # prepare adversarial embeddings
        adv_embeds = self.adv_embeds if adv_mask is not None else None
        if adv_embeds is not None and adv_embeds.size(0) == 1:
            adv_embeds = adv_embeds.expand(input_ids.size(0), -1, -1)

        inputs_embeds = self.adv_embedder.forward(input_ids, adv_embeds, adv_mask)

        return self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        adv_mask: torch.Tensor | None = None,
        max_length: int = 100,
        **kwargs,
    ) -> GenerateDecoderOnlyOutput:

        # prepare adversarial embeddings
        adv_embeds = self.adv_embeds if adv_mask is not None else None
        if adv_embeds is not None and adv_embeds.size(0) == 1:
            adv_embeds = adv_embeds.expand(input_ids.size(0), -1, -1)

        inputs_embeds = self.adv_embedder.forward(input_ids, adv_embeds, adv_mask)

        # greedy by default
        kwargs.setdefault("do_sample", False)
        kwargs.setdefault("temperature", 1.0)
        kwargs.setdefault("top_p", 1.0)
        kwargs.setdefault("top_k", None)

        return self.model.generate(
            inputs=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_model_defaults=True,
            **kwargs,
        )  # type: ignore

    @torch.inference_mode()
    def chat(
        self,
        conversations: list[list[dict[str, str]]],
        max_length: int = 100,
        **kwargs,
    ) -> list[str]:
        """
        Generates adversarial texts for a batch of conversations.

        Args:
            conversations (list[list[dict[str, str]]]): A batch of conversations, where each conversation is a list of messages.
                Each message is a dictionary with keys "role" and "content".
            max_length (int): Maximum length of the generated text.

        Returns:
            list[str]: List of generated adversarial texts.
        """

        token_dict = self.tokenize(conversations)

        result = self.generate(
            input_ids=token_dict["input_ids"],
            attention_mask=token_dict["attention_mask"],
            adv_mask=token_dict["adv_mask"],
            max_length=max_length,
            **kwargs,
        )

        return self.tokenizer.batch_decode(result, skip_special_tokens=True)
