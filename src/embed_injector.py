import torch
from torch import nn
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm.auto import tqdm
import abc

from src import utils


class EmbedInjector(nn.Module):
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
            self.model.resize_token_embeddings(len(tokenizer))

        # params
        self.device = utils.extract_device(model)
        self.num_tokens = num_tokens

    @property
    def embed_dim(self) -> int:
        """
        Returns the embedding dimension of the model.
        """
        return self.model.get_input_embeddings().weight.size(-1)

    def tokenize(
        self,
        input_texts: list[str],
        target_texts: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize the input and target texts.
        This function pads the input and target texts such that the adversarial tokens are aligned across all samples.

        Args:
            input_texts (list[str]): List of input texts.
            target_texts (list[str] | None): List of target texts. If None, only input texts are tokenized.

        Returns:
            dict: Dictionary containing the tokenized input and target texts, with the following keys:
                - input_ids: Token IDs of the entire tokenized texts.
                - attention_mask: Attention mask of the entire tokenized texts.
                - adv_mask: Mask for the adversarial tokens.
                - const_idx: Mask for the constant tokens for KV-cache.
                - target_mask: Mask for the target tokens, if provided.
        """

        input_messeges = []
        for inp_txt in input_texts:
            msg = [{"role": "user", "content": inp_txt + (self.adv_token * self.num_tokens)}]
            input_messeges.append(msg)

        self.tokenizer.padding_side = "left"
        input_tokens = self.tokenizer.apply_chat_template(
            input_messeges,
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
            # - samwit/koala-7b - target should begin with <think> token

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

    def embed(
        self,
        inputs: list[str],
        targets: list[str] | None = None,
    ) -> dict:
        """
        Embed the input and target texts using the model's input embeddings.

        Args:
            inputs (list[str]): List of input texts.
            targets (list[str] | None): List of target texts. If None, only input texts are embedded.

        Returns:
            dict[str,torch.Tensor]: Dictionary containing the embedded input and target texts, with the following keys:
                - input_ids: Token IDs of the entire tokenized texts.
                - attention_mask: Attention mask of the entire tokenized texts.
                - adv_mask: Mask for the adversarial tokens.
                - const_idx: Per sample index of the constant tokens, for KV-caching.
                - target_mask: Mask for the target tokens, if provided.
                - inputs_embeds: Input embeddings of the entire tokenized texts.
        """
        tokenize_result = self.tokenize(inputs, targets)
        embedder = self.model.get_input_embeddings()
        inputs_embeds = embedder(tokenize_result["input_ids"])
        tokenize_result["inputs_embeds"] = inputs_embeds
        return tokenize_result

    def inject_embed(
        self,
        inputs_embeds: torch.Tensor,
        adver_embeds: torch.Tensor,
        adver_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject the adversarial embedding into the input embeddings.
        """
        assert inputs_embeds.ndim == adver_embeds.ndim
        assert inputs_embeds.size(0) == adver_embeds.size(0) == adver_mask.size(0)  # same batch size
        assert inputs_embeds.size(-1) == adver_embeds.size(-1)  # same embedding size

        if inputs_embeds.ndim == adver_mask.ndim + 1:
            adver_mask = adver_mask.unsqueeze(-1)

        return inputs_embeds.masked_scatter(mask=adver_mask, source=adver_embeds)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attn_mask: torch.Tensor,
        **kwargs,
    ):
        return self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, input_texts: list[str], adv_embed: torch.Tensor, max_length: int = 100) -> list[str]:
        """
        Generate adversarial text using the fitted model.

        Args:
            input_texts (list[str]): List of input texts.
            adv_embed (torch.Tensor): Adversarial embedding.
            max_length (int): Maximum length of the generated text.

        Returns:
            list[str]: List of generated adversarial texts.
        """
        token_dict = self.embed(input_texts)

        inj_embeds = self.inject_embed(
            inputs_embeds=token_dict["inputs_embeds"],
            adver_embeds=adv_embed,
            adver_mask=token_dict["adv_mask"],
        )

        result = self.model.generate(
            input_ids=None,
            inputs_embeds=inj_embeds,
            attention_mask=token_dict["attention_mask"],
            do_sample=True,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return self.tokenizer.batch_decode(result, skip_special_tokens=True)
