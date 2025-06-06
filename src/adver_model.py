import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Iterator
import copy

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
        orig_embedder = self.model.get_input_embeddings()
        self.adv_embedder = AdverEmbedding(orig_embedder)
        self.model.set_input_embeddings(self.adv_embedder)

        # params
        self.device = utils.extract_device(model)
        self.num_tokens = num_tokens

        # adversarial embeddings
        self.adv_embeds: torch.Tensor | None = torch.zeros(
            (1, num_tokens, self.adv_embedder.embed_dim),
            dtype=self.adv_embedder.embed_dtype,
            device=self.device,
        )

    def parameters(self) -> Iterator[nn.Parameter]:
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

    def set_embeddings(self, adv_embeds: torch.Tensor):
        """
        Set the adversarial embeddings.
        If batch_size is 1, the embeddings will be broadcasted to input batch size.

        Args:
            adv_embeds (torch.Tensor): Adversarial embeddings of shape (batch_size, num_tokens, embed_dim).
        """
        self.adv_embeds = adv_embeds

    def get_embeddings(self, clone: bool = False) -> torch.Tensor:
        """
        Get the adversarial embeddings tensor.
        It is highly recommended to get the embedding throught this method.

        Args:
            clone (bool): If True, returns a cloned and detached tensor, otherwise returns the original tensor.

        Returns:
            torch.Tensor: Adversarial embeddings of shape (batch_size, num_tokens, embed_dim).
        """
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

        # add adversarial tokens to the last message in each conversation
        for conv in conversations:
            if len(conv) == 0:
                continue
            last_msg = conv[-1]
            if "content" not in last_msg:
                continue
            last_msg["content"] += (self.adv_token * self.num_tokens)

        return conversations
        
    
    def tokenize(
        self,
        conversations: list[list[dict[str, str]]],
        target_texts: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize the input and target texts.
        This function pads the input and target texts such that the adversarial tokens are aligned across all samples.

        Args:
            conversations (list[list[dict[str, str]]]): A batch of conversations, where each conversation is a list of messages.
                Each message is a dictionary with keys "role" and "content".
            target_texts (list[str] | None): List of target texts. If None, only input texts are tokenized.

        Returns:
            dict: Dictionary containing the tokenized input and target texts, with the following keys:
                - input_ids: Token IDs of the entire tokenized texts.
                - attention_mask: Attention mask of the entire tokenized texts.
                - adv_mask: Mask for the adversarial tokens.
                - const_idx: Mask for the constant tokens for KV-cache.
                - target_mask: Mask for the target tokens, if provided.
        """
        
        conversations = self.inject_tokens(conversations)

        self.tokenizer.padding_side = "left"
        input_tokens = self.tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            padding=True,
            padding_side="left",
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.device)

        token_ids = input_tokens["input_ids"]
        attn_mask = input_tokens["attention_mask"]

        if target_texts is not None:

            # NOTE:
            # we pad from input and target side, such that the adv tokens are aligned across all samples
            # this allows us to use KV-cache efficiently for all samples, and ease of access to adv embedding

            # example (P - padding, I - input, A - adv, T - target):
            # [P][P][P][I][I] [A][A][A] [T][T][P][P]
            # [P][I][I][I][I] [A][A][A] [T][T][T][T]
            # [I][I][I][I][I] [A][A][A] [T][P][P][P]

            # tested on:
            # - meta-llama/Llama-3.2-1B-Instruct
            # - Qwen/Qwen3-0.6B

            self.tokenizer.padding_side = "right"
            target_tokens = self.tokenizer(
                target_texts,
                padding=True,
                padding_side="right",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)

            # check if BOS was added to target, if yes remove it
            if self.tokenizer.bos_token and target_tokens["input_ids"][0][0] == self.tokenizer.bos_token_id:
                target_tokens["input_ids"] = target_tokens["input_ids"][:, 1:]
                target_tokens["attention_mask"] = target_tokens["attention_mask"][:, 1:]

            # combine input and target tokens
            token_ids = torch.cat([token_ids, target_tokens["input_ids"]], dim=1)
            attn_mask = torch.cat([attn_mask, target_tokens["attention_mask"]], dim=1)

            # create target mask
            target_mask = torch.zeros_like(token_ids, dtype=torch.bool)
            target_mask[:, -target_tokens["input_ids"].shape[1] :] = True
            target_mask = torch.logical_and(target_mask, attn_mask == 1)

        # create adv token mask
        adv_token_id = self.tokenizer.convert_tokens_to_ids(self.adv_token)
        adv_mask = token_ids == adv_token_id

        # create const idx, parts of the input batch that does not change
        const_idx = torch.argmax(adv_mask.int(), dim=1)

        result_dict = {
            "input_ids": token_ids,
            "attention_mask": attn_mask,
            "adv_mask": adv_mask,
            "const_idx": const_idx,
        }

        if target_texts is not None:
            result_dict["target_mask"] = target_mask

        return result_dict

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
    ):            

        # prepare adversarial embeddings
        adv_embeds = self.adv_embeds if adv_mask is not None else None
        if adv_embeds is not None and adv_embeds.size(0) == 1:
            adv_embeds = adv_embeds.expand(input_ids.size(0), -1, -1)

        inputs_embeds = self.adv_embedder.forward(input_ids, adv_embeds, adv_mask)

        return self.model.generate(
            inputs=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )

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
