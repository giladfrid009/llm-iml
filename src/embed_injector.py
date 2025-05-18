import torch
from torch import nn
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm.auto import tqdm
import abc

import utils


class EmbedInjector(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        num_tokens: int = 10,
        adv_token: str = "[ADV]",
    ):
        # model
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.device = extract_device(model)

        # tokenizer
        self.tokenizer = tokenizer
        self.adv_token = adv_token
        if self.adv_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [self.adv_token]})
            self.model.resize_token_embeddings(len(tokenizer))
            
        # attack params
        self.num_tokens = num_tokens


    def tokenize_input_target(self, input_texts: list[str], target_texts: list[str]):
        """
        Tokenize the input and target texts.
        This function pads the input and target texts such that the adversarial tokens are aligned across all samples.

        Args:
            input_texts (list[str]): List of input texts.
            target_texts (list[str]): List of target texts.

        Returns:
            dict: Dictionary containing the tokenized input and target texts, with the following keys:
                - input_ids: Token IDs of the input texts.
                - attention_mask: Attention mask for the input texts.
                - adv_mask: Mask for the adversarial tokens.
                - target_mask: Mask for the target texts.
        """

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

        input_messeges = []
        for inp_txt in input_texts:
            msg = [{"role": "user", "content": inp_txt + (self.adv_token * self.num_tokens)}]
            input_messeges.append(msg)

        tokenizer.padding_side = "left"
        input_tokens = self.tokenizer.apply_chat_template(
            input_messeges,
            add_generation_prompt=True,
            padding=True,
            padding_side="left",
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.device)

        tokenizer.padding_side = "right"
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
        token_ids = torch.cat([input_tokens["input_ids"], target_tokens["input_ids"]], dim=1)
        attn_mask = torch.cat([input_tokens["attention_mask"], target_tokens["attention_mask"]], dim=1)

        # create adv token mask
        adv_token_id = self.tokenizer.convert_tokens_to_ids(self.adv_token)
        adv_mask = token_ids == adv_token_id

        # create target mask
        target_mask = torch.zeros_like(token_ids, dtype=torch.bool, device=self.device)
        target_mask[:, -target_tokens["input_ids"].shape[1] :] = True
        target_mask = torch.logical_and(target_mask, attn_mask == 1)

        return {
            "input_ids": token_ids,
            "attention_mask": attn_mask,
            "adv_mask": adv_mask,
            "target_mask": target_mask,
        }


    def tokenize_input(self, input_texts: list[str]):
        """
        Tokenize the input texts.

        Args:
            input_texts (list[str]): List of input texts.

        Returns:
            dict: Dictionary containing the tokenized input texts with the following keys:
                - input_ids: Token IDs of the input texts.
                - attention_mask: Attention mask for the input texts.
                - adv_mask: Mask for the adversarial tokens.

        """
        input_messeges = []
        for inp_txt in input_texts:
            msg = [
                {"role": "user", "content": inp_txt + (self.adv_token * self.num_tokens)},
            ]
            input_messeges.append(msg)

        tokenizer.padding_side = "left"
        input_tokens = self.tokenizer.apply_chat_template(
            input_messeges,
            add_generation_prompt=True,
            padding=True,
            padding_side="left",
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.device)

        # create adv token mask
        adv_token_id = self.tokenizer.convert_tokens_to_ids(self.adv_token)
        adv_mask = input_tokens["input_ids"] == adv_token_id

        return {
            "input_ids": input_tokens["input_ids"],
            "attention_mask": input_tokens["attention_mask"],
            "adv_mask": adv_mask,
        }
        
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
        token_dict = self.tokenize_input(input_texts)
        inputs_embeds = self.embed(token_dict["input_ids"])
        inputs_embeds = inputs_embeds.masked_scatter(mask=token_dict["adv_mask"].unsqueeze(-1), source=adv_embed)

        result = self.model.generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=token_dict["attention_mask"],
            do_sample=True,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return self.tokenizer.batch_decode(result, skip_special_tokens=True)