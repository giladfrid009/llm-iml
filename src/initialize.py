import logging
import torch
from src.adver_model import AdverModel


logger = logging.getLogger(__name__)


class Initializer:
    """
    A class used to initialize universal adversarial embeddings for a given adversarial model..
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
    def from_string(
        adver_model: AdverModel,
        text: str,
        strict: bool = True,
        pad_word: str = " ",
    ):
        """
        Initialize adversarial embeddings from a string of text.
        
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


    # TODO: implement sampling from mean and std of all embeddings
    # TODO: maybe implement sampling from mean and same covariance (as done in HF transformers)
    # TODO: maybe implement sampling from Lp ball

    @staticmethod
    def _sample_lp_ball(length: int, norm: float = 2.0, device: torch.device | None = None) -> torch.Tensor:
        """
        Implementation of the method described in:
        [https://stats.stackexchange.com/questions/352668/generate-uniform-noise-from-a-p-norm-ball-x-p-leq-r]

        Sample a random vector from the Lp ball of radius 1.

        Args:
            length (int): Length of the vector.
            norm (int): Norm of the ball.
            device (torch.device, optional): Device to move the tensor to.

        Returns:
            torch.Tensor: A random vector sampled from the Lp ball.
        """
        if device is None:
            device = torch.device("cpu")
        vec = (-torch.log(torch.rand(length, device=device))) ** (1 / norm)
        sgn = 2 * torch.randint(0, 2, (length,), dtype=torch.float32, device=device) - 1
        vec = sgn * vec
        vec = vec / (torch.norm(vec, p=norm) + torch.finfo(vec.dtype).eps)
        rad = torch.exp(torch.log(torch.rand(1, device=device)) / length)
        return rad * vec
