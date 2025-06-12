import logging
import torch
from src.adver_model import AdverModel


logger = logging.getLogger(__name__)


class Initializer:
    """
    A class used to initialize universal adversarial embeddings for a given adversarial model.
    Note, that all these methods set the embedding of the passed :class:`AdverModel` instance.
    """

    @staticmethod
    def make_empty(adver_model: AdverModel) -> torch.Tensor:
        """
        Initialize an empty tensor for adversarial embeddings.
        This is useful for starting with a neutral state.
        """
        return torch.empty(
            size=(1, adver_model.num_tokens, adver_model.adv_embedder.embed_dim),
            dtype=adver_model.adv_embedder.embed_dtype,
            device=adver_model.adv_embedder.device,
        )

    @staticmethod
    def full(adver_model: AdverModel, value: float = 0.0):
        embeds = Initializer.make_empty(adver_model)
        embeds.fill_(value)
        adver_model.set_embeddings(embeds)

    @staticmethod
    def uniform(adver_model: AdverModel, low: float = -1.0, high: float = 1.0):
        embeds = Initializer.make_empty(adver_model)
        embeds.uniform_(low, high)
        adver_model.set_embeddings(embeds)

    @staticmethod
    def normal(
        adver_model: AdverModel,
        mean: float | torch.Tensor = 0.0,
        std: float | torch.Tensor = 1.0,
    ):
        embeds = Initializer.make_empty(adver_model)
        embeds = embeds.normal_() * std + mean
        adver_model.set_embeddings(embeds)

    @staticmethod
    def from_mean_std(adver_model: AdverModel):
        """
        Initialize adversarial embeddings using the mean and standard deviation of the original embeddings.
        The mean and std are computed per embedding dimension across all original embeddings.
        """
        orig_embeds = adver_model.orig_embedder.weight
        mean = orig_embeds.mean(dim=0)
        std = orig_embeds.std(dim=0)

        new_embeds = Initializer.make_empty(adver_model)
        new_embeds = new_embeds.normal_() * std + mean
        adver_model.set_embeddings(new_embeds)

    @staticmethod
    def from_lp_ball(
        adver_model: AdverModel,
        norm: float = 2.0,
        radius: float = 1.0,
    ):
        """
        Initialize adversarial embeddings by sampling each token-embedding
        uniformly from the Lp ball of given norm and radius.

        Args:
            adver_model (AdverModel): The adversarial model to initialize.
            norm (float): The norm of the Lp ball to sample from.
            radius (float): The radius of the Lp ball to sample from.
        """
        embeds = Initializer.make_empty(adver_model)
        device = embeds.device
        N, D = adver_model.num_tokens, embeds.shape[-1]

        vec = torch.rand(N, D, device=device)
        vec = (-vec.log()).pow(1.0 / norm)
        sgn = torch.randint(0, 2, (N, D), device=device, dtype=embeds.dtype) * 2 - 1
        vec = sgn * vec
        vec = vec / (vec.norm(p=norm, dim=-1, keepdim=True) + torch.finfo(embeds.dtype).eps)
        rad = torch.rand((N, 1), device=device).pow(1.0 / D) * radius

        embeds = vec * rad
        embeds = embeds.unsqueeze(0)  # add batch dimension
        adver_model.set_embeddings(embeds)

    @staticmethod
    def from_string(
        adver_model: AdverModel,
        text: str,
        strict: bool = True,
        pad_word: str = " ",
    ):
        """
        Initialize adversarial embeddings from a string of text.
        This method tokenizes the text and uses the original embedder to create embeddings.

        Args:
            adver_model (AdverModel): The adversarial model to initialize.
            text (str): The text to use for initialization.
            strict (bool): Whether to enforce a strict length for the embeddings.
                - If True, the embeddings will be padded or truncated to match :attr:`adver_model.num_tokens`.
                - If False, allow variable length. That may result in a change of :attr:`adver_model.num_tokens` value.
            pad_word (str): The word to use for padding if strict is True, to reach the target length.
        """
        tokenizer = adver_model.tokenizer
        embedder = adver_model.orig_embedder

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
            )

            input_ids = tokenized["input_ids"]
            embeddings = embedder(input_ids).unsqueeze(0)
            adver_model.set_embeddings(embeddings, strict=False)
            return

        # Otherwise, we need to ensure the embeddings are of the correct length
        target_length = adver_model.num_tokens
        strict_tokenized = tokenizer.__call__(
            test=text,
            add_special_tokens=False,
            padding="max_length",
            truncation="max_length",
            padding_side="right",
            truncation_side="right",
            max_length=target_length,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_ids = strict_tokenized["input_ids"]
        attention_mask = strict_tokenized["attention_mask"]

        # replace padding tokens with the specified pad_word
        pad_token_id = tokenizer.convert_tokens_to_ids(pad_word)
        input_ids[attention_mask == 0] = pad_token_id

        embeddings = embedder(input_ids).unsqueeze(0)
        adver_model.set_embeddings(embeddings)

    # TODO: maybe implement sampling from mean and same covariance (as done in HF transformers)
