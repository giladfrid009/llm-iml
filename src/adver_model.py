import torch
from torch import nn
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm.auto import tqdm

from src import utils


class AdverEmbedding(nn.Module):
    def __init__(self, embedder: nn.Module):
        super().__init__()
        self.embedder = embedder
        self.device = utils.extract_device(embedder)

        self.adv_embeds = None
        self.adv_mask = None

    def _verify(self, adv_emb: torch.Tensor | None, adv_mask: torch.Tensor | None):
        if (adv_emb is None) != (adv_mask is None):
            raise ValueError("Both adv_emb and adv_mask should be None or not None")

    def embed_dim(self) -> int:
        test_input = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        return self.embedder(test_input).size(-1)

    def embed_dtype(self) -> torch.dtype:
        test_input = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        return self.embedder(test_input).dtype

    def set_adver(self, adv_embeds: torch.Tensor | None = None, adv_mask: torch.Tensor | None = None):
        self._verify(adv_embeds, adv_mask)
        self.adv_embeds = adv_embeds
        self.adv_mask = adv_mask

    def forward(
        self,
        input: torch.Tensor,
        adv_embeds: torch.Tensor | None = None,
        adv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._verify(adv_embeds, adv_mask)

        if adv_embeds is None:  # use the stored values
            adv_embeds = self.adv_embeds
            adv_mask = self.adv_mask

        embedded: torch.Tensor = self.embedder(input)

        if adv_embeds is None:
            return embedded

        assert embedded.ndim == adv_embeds.ndim  # same number of dimensions
        assert embedded.size(0) == adv_embeds.size(0)  # same batch size
        assert embedded.size(-1) == adv_embeds.size(-1)  # same embedding size

        if adv_mask.ndim == adv_embeds.ndim - 1:
            adv_mask = adv_mask.unsqueeze(-1)  # we assume the mask is over the tokens

        return embedded.masked_scatter(mask=adv_mask, source=adv_embeds)


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
            model.resize_token_embeddings(len(tokenizer))

        # adv embedder
        orig_embedder = self.model.get_input_embeddings()
        self.adv_embedder = AdverEmbedding(orig_embedder)
        self.model.set_input_embeddings(self.adv_embedder)
        self.embed_dim = self.adv_embedder.embed_dim()
        self.embed_dtype = self.adv_embedder.embed_dtype()

        # params
        self.device = utils.extract_device(model)
        self.num_tokens = num_tokens

    def tokenize(
        self,
        input_texts: list[str],
        target_texts: list[str] | None = None,
        system_texts: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize the input and target texts.
        This function pads the input and target texts such that the adversarial tokens are aligned across all samples.

        Args:
            input_texts (list[str]): List of input texts.
            target_texts (list[str] | None): List of target texts. If None, only input texts are tokenized.
            system_texts (list[str] | None): List of system prompts. If None, default system prompt is used.

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
            
        if system_texts is not None:
            for msg, sys_txt in zip(input_messeges, system_texts):
                msg.insert(0, {"role": "system", "content": sys_txt})

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
        adv_embeds: torch.Tensor | None = None,
        adv_mask: torch.Tensor | None = None,
        **kwargs,
    ):

        self.adv_embedder.set_adver(adv_embeds, adv_mask)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        adv_embeds: torch.Tensor | None = None,
        adv_mask: torch.Tensor | None = None,
        max_length: int = 100,
        **kwargs,
    ):
        self.adv_embedder.set_adver(adv_embeds, adv_mask)

        if adv_embeds is None:
            # standard generation
            return self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

        else:
            # adversarial generation
            inputs_embeds = self.adv_embedder.forward(input_ids, adv_embeds, adv_mask)
            self.adv_embedder.set_adver(None, None)  # reset the adv embeds and mask
            return self.model.generate(
                inputs=None, # NOTE: if set to input_ids then result will also contain the input prompt
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=True,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

    @torch.inference_mode()
    def generate_text(
        self,
        input_texts: list[str],
        adv_embeds: torch.Tensor,
        max_length: int = 100,
        system_texts: list[str] | None = None,
        **kwargs,
    ) -> list[str]:
        """
        Generate adversarial text using the fitted model.

        Args:
            input_texts (list[str]): List of input texts.
            adv_embeds (torch.Tensor): Adversarial embedding.
            max_length (int): Maximum length of the generated text.
            system_texts (list[str] | None): List of system prompts. If None, default system prompt is used.

        Returns:
            list[str]: List of generated adversarial texts.
        """
        token_dict = self.tokenize(input_texts, system_texts=system_texts)

        with torch.autocast(device_type=self.device.type, enabled=True):

            result = self.generate(
                input_ids=token_dict["input_ids"],
                attention_mask=token_dict["attention_mask"],
                adv_embeds=adv_embeds,
                adv_mask=token_dict["adv_mask"],
                max_length=max_length,
                **kwargs,
            )

        return self.tokenizer.batch_decode(result, skip_special_tokens=True)
