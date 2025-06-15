SUPPORTED_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "GraySwanAI/Llama-3-8B-Instruct-RR", # NOTE: protected model
    "GraySwanAI/Mistral-7B-Instruct-RR", # NOTE: protected model
    "Orenguteng/Llama-3-8B-Lexi-Uncensored",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    # "lmsys/vicuna-7b-v1.5", # TODO: no chat template
    "mistralai/Mistral-7B-Instruct-v0.3",
    # "mistralai/Mixtral-8x7B-Instruct-v0.1", # NOTE: 49B parameters
    "tiiuae/falcon-7b-instruct",
    # "mosaicml/mpt-7b-chat", # TODO: no chat template
    # "microsoft/Orca-2-7b", # TODO: no chat template
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-4-mini-instruct",
    "upstage/SOLAR-10.7B-Instruct-v1.0",
    "openchat/openchat-3.5-0106",    
    "HuggingFaceH4/zephyr-7b-beta",
    "cais/zephyr_7b_r2d2", # NOTE: protected model
    # "openai-community/gpt2", # TODO: no chat template
    "google/gemma-2b-it",
    "google/gemma-2-2b-it",
    "google/gemma-3-1b-it",
    # "ContinuousAT/Llama-2-7B-CAT" # NOTE: protected model, only an adapter
]

def print_models():
    print("Supported Models:")
    for model in SUPPORTED_MODELS:
        print(f"- {model}")