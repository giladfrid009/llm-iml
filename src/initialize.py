from __future__ import annotations
from typing import TYPE_CHECKING
from tqdm.auto import tqdm
import torch

from src.data import TableLoader
from src.utils.logging import create_logger


if TYPE_CHECKING:
    from src.adv_model import AdvModel
    from src.sample_attacks import SampleAttack


logger = create_logger(__name__)


class Initializer:
    """
    A class used to initialize universal adversarial embeddings for a given adversarial model.
    Note, that all these methods set the embedding of the passed :class:`AdvModel` instance.
    """

    @staticmethod
    def make_empty(adv_model: AdvModel, batch_size: int = 1) -> torch.Tensor:
        """
        Initialize an empty tensor for adversarial embeddings.
        This is useful for starting with a neutral state.

        Args:
            adv_model (AdvModel): The adversarial model to initialize.
            batch_size (int): The batch size for the embeddings tensor. 1 for a universal embedding, > 1 for batch of samples.
        """
        return torch.empty(
            size=(batch_size, adv_model.num_tokens, adv_model.adv_embedder.embed_dim),
            dtype=adv_model.adv_embedder.embed_dtype,
            device=adv_model.adv_embedder.device,
        )

    @staticmethod
    def full(adv_model: AdvModel, value: float = 0.0, batch_size: int = 1) -> torch.Tensor:
        embeds = Initializer.make_empty(adv_model, batch_size=batch_size)
        embeds.fill_(value)
        return embeds

    @staticmethod
    def random_uniform(
        adv_model: AdvModel,
        low: float = -1.0,
        high: float = 1.0,
        batch_size: int = 1,
    ) -> torch.Tensor:
        embeds = Initializer.make_empty(adv_model, batch_size=batch_size)
        embeds.uniform_(low, high)
        return embeds

    @staticmethod
    def random_normal(
        adv_model: AdvModel,
        mean: float = 0.0,
        std: float | None = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        if std is None:
            std = float(adv_model.model.config.initializer_range)
            logger.info(f"Using default std from model config: {std}")

        embeds = Initializer.make_empty(adv_model, batch_size=batch_size)
        embeds = embeds.normal_(mean, std)
        return embeds

    @staticmethod
    def from_mean(
        adv_model: AdvModel,
        std: float | None = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Initialize adversarial embeddings using the mean of the original embeddings.
        The mean is computed per embedding dimension across all original embeddings.
        """
        if std is None:
            std = float(adv_model.model.config.initializer_range)
            logger.info(f"Using default std from model config: {std}")

        with torch.no_grad():
            orig_weight = adv_model.orig_embedder.weight.float()
            mean = torch.mean(orig_weight, dim=0, keepdim=True)

        embeds = Initializer.make_empty(adv_model, batch_size=batch_size)
        embeds = embeds.normal_(std=std) + mean
        embeds = embeds.detach().to(adv_model.adv_embedder.embed_dtype)
        return embeds

    @staticmethod
    def from_mean_std(adv_model: AdvModel, batch_size: int = 1) -> torch.Tensor:
        """
        Initialize adversarial embeddings using the mean and standard deviation of the original embeddings.
        The mean and std are computed per embedding dimension across all original embeddings.
        """
        with torch.no_grad():
            orig_weight = adv_model.orig_embedder.weight.float()
            mean = torch.mean(orig_weight, dim=0, keepdim=True)
            std = torch.std(orig_weight, dim=0, keepdim=True)

        embeds = Initializer.make_empty(adv_model, batch_size=batch_size)
        embeds = embeds.normal_() * std + mean
        embeds = embeds.detach().to(adv_model.adv_embedder.embed_dtype)
        return embeds

    @staticmethod
    def from_covariance(
        adv_model: AdvModel,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Same initialization as in `transformers.modeling_utils.resize_token_embeddings`, when setting `mean_resizing=True`.

        #### Transformers documentation:
        The generated tokens' probabilities won't be affected by the added embeddings because initializing the new embeddings with the
        old embeddings' mean will reduce the kl-divergence between the next token probability before and after adding the new embeddings.
        Refer to this article for more information: https://nlp.stanford.edu/~johnhew/vocab-expansion.html
        """
        import torch.distributions.constraints as constraints
        from torch.distributions.multivariate_normal import MultivariateNormal

        with torch.no_grad():
            orig_weights = adv_model.orig_embedder.weight.float()
            mean_weights = torch.mean(orig_weights, dim=0)
            centered_weights = orig_weights - mean_weights
            covariance = torch.cov(centered_weights.T, correction=1)

        if constraints.positive_definite.check(covariance).all():
            dist = MultivariateNormal(mean_weights, covariance_matrix=covariance)
            embeds = dist.sample(sample_shape=(batch_size, adv_model.num_tokens))
            return embeds.to(adv_model.adv_embedder.embed_dtype)

        logger.info("Covariance matrix is not PD. Falling back to mean initialization.")
        return Initializer.from_mean_std(adv_model, batch_size=batch_size)

    @staticmethod
    def from_string(
        adv_model: AdvModel,
        text: str,
        strict: bool = True,
        pad_word: str = ".",
        verbose: bool = True,
        batch_size: int = 1,
    ) -> torch.Tensor:
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
            batch_size (int): The batch size for the embeddings tensor. 1 for a universal prompt, > 1 for batch of prompts.
        """
        tokenizer = adv_model.tokenizer
        input_ids: torch.Tensor

        if not strict:
            # Tokenize without padding or truncation, which may
            # modify the number of adversarial tokens
            encodings = tokenizer(
                text,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_tensors="pt",
                return_attention_mask=False,
            ).to(adv_model.device)

            input_ids = encodings.input_ids

        else:
            # We need to set some tokenizer settings manually
            orig_padding_side = tokenizer.padding_side
            orig_truncation_side = tokenizer.truncation_side
            tokenizer.truncation_side = "right"
            tokenizer.padding_side = "right"

            # Tokenize with strict padding and truncation
            strict_encodings = tokenizer(
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

            input_ids = strict_encodings.input_ids
            attention_mask = strict_encodings.attention_mask

            # replace padding tokens with the specified pad_word
            pad_token_id: int = tokenizer.convert_tokens_to_ids(pad_word)  # type: ignore
            input_ids[attention_mask == 0] = pad_token_id

        with torch.no_grad():
            embeds = adv_model.orig_embedder.forward(input_ids)

        if batch_size > 1:
            embeds = embeds.repeat(batch_size, 1, 1)

        if verbose:
            ids_list = input_ids.flatten().tolist()
            str_list = tokenizer.convert_ids_to_tokens(ids_list, skip_special_tokens=False)
            logger.info(f"Initialized from text: '{text}'")
            logger.info(f"Embed Tokens: {str_list}")
            logger.info(f"Embed Length: {len(str_list)}")

        return embeds.detach()

    @staticmethod
    def from_random_ids(
        adv_model: AdvModel,
        allow_nonascii: bool = False,
        allow_special: bool = False,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Initialize adversarial embeddings from random token IDs.
        This method samples random token IDs from the tokenizer's vocabulary,
        according to the specified constraints, and uses the original embedder to create embeddings.

        Args:
            adv_model (AdvModel): The adversarial model to initialize.
            allow_nonascii (bool): Whether to allow non-ascii tokens.
            allow_special (bool): Whether to allow special tokens.
            batch_size (int): The batch size for the embeddings tensor. 1 for a universal prompt, > 1 for batch of prompts.
        """
        tokenizer = adv_model.tokenizer

        def is_ascii(id: int) -> bool:
            s = tokenizer.convert_ids_to_tokens(id)
            return s.isascii() and s.isprintable()

        allowed_ids = list(range(adv_model.orig_embedder.num_embeddings))

        if not allow_nonascii:
            allowed_ids = [id for id in allowed_ids if is_ascii(id)]

        if not allow_special:
            allowed_ids = [id for id in allowed_ids if id not in tokenizer.all_special_ids]

        allowed_ids = torch.tensor(allowed_ids, device=adv_model.device)

        rand_indices = torch.randint(
            low=0,
            high=len(allowed_ids),
            size=(batch_size, adv_model.num_tokens),
            device=adv_model.device,
        )

        with torch.no_grad():
            rand_ids = allowed_ids[rand_indices]
            embeds = adv_model.orig_embedder.forward(rand_ids)
            return embeds.detach()

    @staticmethod
    def load(
        adv_model: AdvModel,
        path: str,
        strict: bool = True,
        batch_size: int = 1,
    ) -> torch.Tensor:
        embeds: torch.Tensor = torch.load(path, map_location=adv_model.adv_embedder.device, weights_only=True)

        if embeds.ndim != 3:
            raise ValueError(f"Loaded embeddings must be a 3D tensor, got shape: {embeds.shape}.")

        B, N, H = batch_size, adv_model.num_tokens, adv_model.adv_embedder.embed_dim
        eB, eN, eH = embeds.size(0), embeds.size(1), embeds.size(2)

        if eH != H:
            raise ValueError(f"Loaded embeddings hidden dimension mismatch with adv_embedder.embed_dim ({eH} != {H}).")

        if eB != B:
            raise ValueError(f"Loaded embeddings batch size mismatch with requested batch size ({eB} != {B}).")

        if strict and eN != N:
            raise ValueError(f"Loaded embeddings length mismatch with adv_model.num_tokens ({eN} != {N}).")

        elif not strict and eN != N:
            logger.info(f"Loaded embeddings length mismatch with adv_model.num_tokens ({eN} != {N}).")

        return embeds.to(adv_model.adv_embedder.embed_dtype)

    @staticmethod
    def CRI(
        adv_model: AdvModel,
        candidates: list[torch.Tensor],
        test_prompts: list[str],
        test_targets: list[str],
    ) -> torch.Tensor:
        """
        Initialize adversarial embeddings using Compliance Refusal Initialization (CRI) method.
        The candidate which minimizes the loss on the provided test prompts and targets is selected.
        Reference: [https://arxiv.org/pdf/2502.09755].

        Args:
            adv_model (AdvModel): The adversarial model to initialize.
            candidates (list[torch.Tensor]): A list of candidate adversarial embeddings to evaluate.
            test_prompts (list[str]): A list of test prompts to evaluate the candidates.
            test_targets (list[str]): A list of target responses corresponding to the test prompts.

        Returns:
            torch.Tensor: The best candidate adversarial embeddings.
        """
        if len(candidates) == 0:
            raise ValueError("candidates list must contain at least one tensor.")

        conversations = [[{"role": "user", "content": prompt}] for prompt in test_prompts]
        conversations = adv_model.inject_tokens(conversations)
        encodings = adv_model.tokenize(conversations, test_targets)
        criterion = torch.nn.CrossEntropyLoss()

        lowest_loss = float("inf")
        best_embeds = candidates[0]

        with tqdm(candidates, desc="Evaluating candidates", total=len(candidates)) as pbar:
            for candidate_embeds in pbar:
                with torch.inference_mode():
                    result = adv_model.forward(
                        input_ids=encodings.input_ids,
                        attention_mask=encodings.attention_mask,
                        adv_mask=encodings.adv_mask,
                        adv_embeds=candidate_embeds,
                    )

                # align predicted logits and target_ids
                logits = result.logits[:, :-1]  # remove new token
                input_ids = encodings.input_ids[:, 1:]  # remove BOS token
                target_mask = encodings.target_mask[:, 1:]  # remove BOS token

                # extract only targets
                target_logits = logits[target_mask].view(-1, logits.size(-1))
                target_ids = input_ids[target_mask].view(-1)

                # compute token-wise loss
                loss = criterion.forward(target_logits, target_ids).item()

                if loss < lowest_loss:
                    lowest_loss = loss
                    best_embeds = candidate_embeds

                pbar.set_postfix(loss=loss, best_loss=lowest_loss)

        return best_embeds  # type: ignore

    @staticmethod
    def sampleCRI(
        adv_model: AdvModel,
        sample_attack: SampleAttack,
        dataloader: TableLoader,
        num_candidates: int = 100,
    ) -> torch.Tensor:
        """
        Initialize adversarial embeddings using Sample-Attack based Compliance Refusal Initialization (CRI) method.
        Candidates for CRI are generated using the provided Sample-Attack instance and the given dataloader.
        The candidate which minimizes the loss on a random batch from the dataloader is selected.

        Args:
            adv_model (AdvModel): The adversarial model to initialize.
            sample_attack (SampleAttack): The Sample-Attack instance used to generate candidates.
            dataloader (TableLoader): A dataloader providing data for generating candidates.
            num_candidates (int): The number of candidates to generate and evaluate.
        """

        assert sample_attack.adv_model == adv_model, "SampleAttack's adv_model must match the provided adv_model."

        # make sure the dataloader is shuffled
        dataloader = dataloader.copy(shuffle=True)

        # a random batch is used for testing candidates
        batch0 = next(iter(dataloader))
        test_prompts = batch0["prompt"]
        test_targets = batch0["target"]

        candidate_list = []

        with tqdm(total=num_candidates, desc="Sampling candidates") as pbar:
            while True:
                for batch in dataloader:
                    convs = [[{"role": "user", "content": prm}] for prm in batch["prompt"]]
                    sample_results = sample_attack.fit(convs, target_texts=batch["target"])

                    if sample_results.adv_embeds is None:
                        raise ValueError(f"{type(sample_attack).__name__} did not produce adversarial embeddings.")

                    # split along batch dimension
                    embeds = torch.unbind(sample_results.adv_embeds, dim=0)
                    embeds = [c.unsqueeze(0) for c in embeds]

                    # update candidate list
                    candidate_list.extend(embeds)
                    pbar.update(len(convs))

                    # if we have enough candidates, select the best one using CRI
                    if len(candidate_list) >= num_candidates:
                        return Initializer.CRI(
                            adv_model,
                            candidates=candidate_list,
                            test_prompts=test_prompts,
                            test_targets=test_targets,
                        )
