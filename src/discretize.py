import torch
import torch.nn.functional as F
from collections import OrderedDict


class Discretize:
    """
    Discretization functions for projecting soft embeddings to hard token ids.
    """
    
    # LRU cache for normalized vocabulary matrices to avoid repeated normalization
    _normalized_vocab_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
    _cache_max_size: int = 10

    @staticmethod
    def cosine_similarity(
        soft_embeds: torch.Tensor,
        vocab_matrix: torch.Tensor,
        forbidden_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Discretize the soft embeddings to the nearest hard embedding using cosine similarity.

        Args:
            soft_embeds (torch.Tensor): Soft embeddings of shape [b, n, d]
            vocab_matrix (torch.Tensor): Vocabulary embeddings of shape [v, d]
            forbidden_mask (torch.Tensor | None): A boolean mask of shape [v] indicating which tokens are forbidden (True) or allowed (False).

        Returns:
            ids (torch.Tensor): Nearest neighbor of each soft embedding in the vocabulary, shape [b, n]
        """

        # L2-normalize query embeddings
        q = F.normalize(soft_embeds, p=2, dim=-1)  # [b, n, d]
        
        # LRU cache for normalized vocabulary to avoid repeated normalization
        vocab_id = id(vocab_matrix)
        if vocab_id not in Discretize._normalized_vocab_cache:
            Discretize._normalized_vocab_cache[vocab_id] = F.normalize(vocab_matrix, p=2, dim=-1)
            # LRU eviction: remove oldest entry when cache is full
            if len(Discretize._normalized_vocab_cache) > Discretize._cache_max_size:
                Discretize._normalized_vocab_cache.popitem(last=False)
        else:
            # Move to end (most recently used)
            Discretize._normalized_vocab_cache.move_to_end(vocab_id)
        
        w = Discretize._normalized_vocab_cache[vocab_id]  # [v, d]

        # cosine sim == dot product for unit vectors
        sims = torch.matmul(q, w.T)  # [b, n, v]
        if forbidden_mask is not None:
            sims = sims.masked_fill(forbidden_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        ids = sims.argmax(dim=-1)  # [b, n]
        return ids
