import logging
import torch
from src.adv_model import AdvModel

logger = logging.getLogger(__name__)


class Initializer:
    """
    A class used to initialize universal adversarial embeddings for a given adversarial model.
    Note, that all these methods set the embedding of the passed :class:`AdvModel` instance.
    """

    @staticmethod
    def make_empty(adv_model: AdvModel) -> torch.Tensor:
        """
        Initialize an empty tensor for adversarial embeddings.
        This is useful for starting with a neutral state.
        """
        return torch.empty(
            size=(1, adv_model.num_tokens, adv_model.adv_embedder.embed_dim),
            dtype=adv_model.adv_embedder.embed_dtype,
            device=adv_model.adv_embedder.device,
        )

    @staticmethod
    def full(adv_model: AdvModel, value: float = 0.0):
        embeds = Initializer.make_empty(adv_model)
        embeds.fill_(value)
        adv_model.set_embeddings(embeds)

    @staticmethod
    def uniform(adv_model: AdvModel, low: float = -1.0, high: float = 1.0):
        embeds = Initializer.make_empty(adv_model)
        embeds.uniform_(low, high)
        adv_model.set_embeddings(embeds)

    @staticmethod
    def normal(
        adv_model: AdvModel,
        mean: float | torch.Tensor = 0.0,
        std: float | torch.Tensor = 1.0,
    ):
        embeds = Initializer.make_empty(adv_model)
        embeds = embeds.normal_() * std + mean
        adv_model.set_embeddings(embeds)

    @staticmethod
    def from_mean_std(adv_model: AdvModel):
        """
        Initialize adversarial embeddings using the mean and standard deviation of the original embeddings.
        The mean and std are computed per embedding dimension across all original embeddings.
        """
        orig_weight: torch.Tensor = adv_model.orig_embedder.weight  # type: ignore
        mean = torch.mean(orig_weight, dim=0)
        std = torch.std(orig_weight, dim=0)

        new_embeds = Initializer.make_empty(adv_model)
        new_embeds = new_embeds.normal_() * std + mean
        adv_model.set_embeddings(new_embeds)

    @staticmethod
    def from_lp_ball(
        adv_model: AdvModel,
        norm: float = 2.0,
        radius: float = 1.0,
    ):
        """
        Initialize adversarial embeddings by sampling each token-embedding
        uniformly from the Lp ball of given norm and radius.

        Args:
            adv_model (AdvModel): The adversarial model to initialize.
            norm (float): The norm of the Lp ball to sample from.
            radius (float): The radius of the Lp ball to sample from.
        """
        embeds = Initializer.make_empty(adv_model)
        device = embeds.device
        N, D = adv_model.num_tokens, embeds.shape[-1]

        vec = torch.rand(N, D, device=device)
        vec = (-vec.log()).pow(1.0 / norm)
        sgn = torch.randint(0, 2, (N, D), device=device, dtype=embeds.dtype) * 2 - 1
        vec = sgn * vec
        vec = vec / (vec.norm(p=norm, dim=-1, keepdim=True) + torch.finfo(embeds.dtype).eps)
        rad = torch.rand((N, 1), device=device).pow(1.0 / D) * radius

        embeds = vec * rad
        embeds = embeds.unsqueeze(0)  # add batch dimension
        adv_model.set_embeddings(embeds)

    @staticmethod
    def from_string(
        adv_model: AdvModel,
        text: str,
        strict: bool = True,
        pad_word: str = ".",
        verbose: bool = False,
    ):
        """
        Initialize adversarial embeddings from a string of text.
        This method tokenizes the text and uses the original embedder to create embeddings.

        Args:
            adv_model (AdvModel): The adversarial model to initialize.
            text (str): The text to use for initialization.
            strict (bool): Whether to enforce a strict length for the embeddings.
                - If True, the embeddings will be padded or truncated to match :attr:`adv_model.num_tokens`.
                - If False, allow variable length. That may result in a change of :attr:`adv_model.num_tokens` value.
            pad_word (str): The word to use for padding if strict is True, to reach the target length.
            verbose (bool): If True, print the initialized tokens and their length.
        """
        tokenizer = adv_model.tokenizer
        embedder = adv_model.orig_embedder
        input_ids = None

        if not strict:
            # Tokenize without padding or truncation, which may
            # modify the numeber of adversarial tokens
            tokenized = tokenizer(
                text,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_tensors="pt",
                return_attention_mask=False,
            ).to(adv_model.device)

            input_ids = tokenized["input_ids"]

        else:
            # We need to set some tokenizer settings manually
            orig_padding_side = tokenizer.padding_side
            orig_truncation_side = tokenizer.truncation_side
            tokenizer.truncation_side = "right"
            tokenizer.padding_side = "right"

            # Tokenize with strict padding and truncation
            strict_tokenized = tokenizer(
                text=text,
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                padding_side="right",
                max_length=adv_model.num_tokens,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(adv_model.device)

            # Restore original tokenizer settings
            tokenizer.padding_side = orig_padding_side
            tokenizer.truncation_side = orig_truncation_side

            input_ids = strict_tokenized["input_ids"]
            attention_mask = strict_tokenized["attention_mask"]

            # replace padding tokens with the specified pad_word
            pad_token_id = tokenizer.convert_tokens_to_ids(pad_word)
            input_ids[attention_mask == 0] = pad_token_id

        embeddings = embedder(input_ids)
        adv_model.set_embeddings(embeddings)

        if verbose:
            ids_list = input_ids.flatten().tolist()
            str_list = tokenizer.convert_ids_to_tokens(ids_list, skip_special_tokens=False)
            print(f"Initialized from text: '{text}'")
            print(f"Embed Tokens: {str_list}")
            print(f"Embed Length: {len(str_list)}")
