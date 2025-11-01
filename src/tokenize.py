import torch
import copy
import re
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from src.aliases import Conv


def chat_with_targets(
    tokenizer: PreTrainedTokenizer,
    conversations: list[Conv],
    target_texts: list[str],
    adv_token: str,
) -> BatchEncoding:
    """
    Tokenization function which also returns a mask indicating positions of target tokens.
    In addition, this function tokenizes the conversations in a kv-cache efficient way.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenization.
        conversations (list[Conv]): A batch of conversations, where each conversation is a list of messages.
        target_texts (list[str]): A list of target texts corresponding to each conversation.
        adv_token (str): The adversarial token to split the conversations on.

    Returns:
        BatchEncoding: A dictionary containing the tokenized input and target texts with the following keys
            - `input_ids` (torch.IntTensor): Token IDs of the entire tokenized texts.
            - `attention_mask` (torch.BoolTensor): Attention mask of the entire tokenized texts.
            - `adv_mask` (torch.BoolTensor): Mask which is true for the adversarial tokens.
            - `const_idx` (torch.LongTensor): The index of the first adversarial token in each conversation.
            - `target_mask` (torch.BoolTensor): A mask indicating the positions of the target tokens in the full input.
    """
    # Optimize: Create modified conversations in-place, avoiding deep copy
    convs_partial = [conv + [{"role": "assistant", "content": ""}] for conv in conversations]
    encodings_partial = chat_with_cache(tokenizer, convs_partial, adv_token)

    convs_full = [conv + [{"role": "assistant", "content": tgt}] for conv, tgt in zip(conversations, target_texts)]
    encodings_full = chat_with_cache(tokenizer, convs_full, adv_token)

    ids_full: torch.Tensor = encodings_full.input_ids
    attn_full: torch.Tensor = encodings_full.attention_mask
    ids_partial: torch.Tensor = encodings_partial.input_ids

    # pad so we can compare the two tensors
    size_diff = ids_full.size(1) - ids_partial.size(1)
    ids_partial = torch.nn.functional.pad(
        ids_partial,
        pad=(0, size_diff),
        value=tokenizer.pad_token_id,  # type: ignore
        mode="constant",
    )

    diff_mask = ids_full != ids_partial
    target_mask = torch.cumsum(diff_mask, dim=1).bool()
    target_mask = torch.logical_and(target_mask, attn_full == 1)

    return BatchEncoding(
        {
            "input_ids": ids_full,
            "attention_mask": attn_full,
            "adv_mask": encodings_full.adv_mask,
            "const_idx": encodings_full.const_idx,
            "target_mask": target_mask,
        }
    )


def chat_with_cache(
    tokenizer: PreTrainedTokenizer,
    conversations: list[Conv],
    adv_token: str,
) -> BatchEncoding:
    """
    Tokenization function which tokenizes the input conversations in a kv-cache efficient way.

    The conversations into two parts:
    1. Constant part: Tokens before the first occurrence of the first adversarial token.
    2. Adversarial part: Tokens including and after the first occurrence of the first adversarial token.

    The split conversation is then padded in a way that the constant part is left-padded
    and the adversarial part is right-padded. This allows for efficient kv-cache usage during training, as the kv-cache for all constant tokens does not change
    even if the embeddings corresponding to the adversarial tokens change during the training.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenization.
        conversations (list[Conv]): A batch of conversations, where each conversation is a list of messages.
        adv_token (str): The adversarial token to split the conversations on.

    Returns:
        BatchEncoding: A dictionary containing the tokenized input and target texts with the following keys
            - `input_ids` (torch.IntTensor): Token IDs of the entire tokenized texts.
            - `attention_mask` (torch.BoolTensor): Attention mask of the entire tokenized texts.
            - `adv_mask` (torch.BoolTensor): Mask which is true for the adversarial tokens.
            - `const_idx` (torch.LongTensor): The index of the first adversarial token in each conversation.
    """
    input_tokens: list[list[int]] = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_special_tokens=True,
        add_generation_prompt=False,
        continue_final_message=True,
        padding=False,
        return_tensors=None,
        return_attention_mask=False,
        return_dict=False,
        enable_thinking=False,
    )  # type: ignore

    adv_token_id: int = tokenizer.convert_tokens_to_ids(adv_token)  # type: ignore

    # find the index of the first adversarial token efficiently
    # Convert to tensor for vectorized operations
    max_len = max(len(conv) for conv in input_tokens)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    # Create padded tensor for efficient searching
    tokens_tensor = torch.full((len(input_tokens), max_len), pad_token_id, dtype=torch.long)
    for i, conv in enumerate(input_tokens):
        tokens_tensor[i, :len(conv)] = torch.tensor(conv, dtype=torch.long)
    
    # Find first occurrence of adv_token_id in each sequence
    adv_mask_temp = tokens_tensor == adv_token_id
    # Get index of first True in each row, or max_len if not found
    const_idx = torch.where(adv_mask_temp.any(dim=1), adv_mask_temp.int().argmax(dim=1), torch.tensor(max_len))
    const_idx = const_idx.tolist()

    # split convs before and after the constant index
    tokens_const = [conv[:idx] for conv, idx in zip(input_tokens, const_idx)]
    tokens_adver = [conv[idx:] for conv, idx in zip(input_tokens, const_idx)]

    # left pad const tokens, and right pad non-const tokens
    # we do that to maximize the effectivness of the kv-cache.
    data_const = tokenizer.pad(
        {"input_ids": tokens_const},
        padding=True,
        padding_side="left",
        return_attention_mask=True,
        return_tensors="pt",
        verbose=False,
    )

    data_adver = tokenizer.pad(
        {"input_ids": tokens_adver},
        padding=True,
        padding_side="right",
        return_attention_mask=True,
        return_tensors="pt",
        verbose=False,
    )

    # construct result tensors
    input_ids = torch.cat([data_const.input_ids, data_adver.input_ids], dim=1)
    attn_mask = torch.cat([data_const.attention_mask, data_adver.attention_mask], dim=1)
    adv_mask = input_ids == adv_token_id
    const_idx = torch.tensor(const_idx, dtype=torch.long)

    return BatchEncoding(
        {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "adv_mask": adv_mask,
            "const_idx": const_idx,
        }
    )


def chat(
    tokenizer: PreTrainedTokenizer,
    conversations: list[Conv],
    adv_token: str,
) -> BatchEncoding:
    """
    Regular chat tokenization function which applies left padding to a batch of conversations.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenization.
        conversations (list[Conv]): A batch of conversations, where each conversation is a list of messages.
        adv_token (str): The adversarial token.

    Returns:
        BatchEncoding: A dictionary containing the tokenized input with the following keys
            - `input_ids` (torch.IntTensor): Token IDs of the entire tokenized texts.
            - `attention_mask` (torch.BoolTensor): Attention mask of the entire tokenized texts.
            - `adv_mask` (torch.BoolTensor): Mask for the adversarial tokens.
            - `const_idx` (torch.LongTensor): The index of the first adversarial token in each conversation.
    """
    tokenizer.padding_side = "left"

    encodings: BatchEncoding = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        padding=True,
        padding_side="left",
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )  # type: ignore

    adv_token_id = tokenizer.convert_tokens_to_ids(adv_token)

    input_ids = encodings.input_ids
    attn_mask = encodings.attention_mask
    adv_mask = input_ids == adv_token_id

    const_idx = torch.argmax(adv_mask.int(), dim=1)
    const_idx[~adv_mask.any(dim=1)] = input_ids.size(1)

    return BatchEncoding(
        {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "adv_mask": adv_mask,
            "const_idx": const_idx,
        }
    )


def replace_tokens(
    tokenizer: PreTrainedTokenizer,
    conversations: list[Conv],
    repl_ids: list[list[int]],
    adv_token: str,
) -> list[Conv]:
    """
    Replace all occurrences of the adversarial token in the conversations with the provided token IDs.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenization.
        conversations (list[Conv]): A batch of conversations, where each conversation is a list of messages.
        repl_ids (list[list[int]]): A list of lists of token IDs to replace the adversarial tokens with.
        adv_token (str): The adversarial token to be replaced.

    Returns:
        list[Conv]: The modified conversations with adversarial tokens replaced by the specified token IDs.
    """
    conversations = copy.deepcopy(conversations)
    pattern = re.compile(re.escape(adv_token))

    for conv, ids in zip(conversations, repl_ids):
        adv_tks = tokenizer.convert_ids_to_tokens(ids)
        adv_it = iter(adv_tks)

        for msg in conv:
            content = msg["content"]
            msg["content"] = pattern.sub(lambda _: next(adv_it), content)

    return conversations
