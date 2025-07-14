from transformers import PreTrainedTokenizer


def get_template(tokenizer: PreTrainedTokenizer, **kwargs) -> str:
    # ======== Else default to tokenizer.apply_chat_template =======
    template = [{"role": "user", "content": "{instruction}"}]
    prompt: str = tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)  # type: ignore
    # Check if the prompt starts with the BOS token
    # removed <s> if it exist (LlamaTokenizer class usually have this) as our baselines will add these if needed later
    if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
        prompt = prompt.replace(tokenizer.bos_token, "")
    return prompt
