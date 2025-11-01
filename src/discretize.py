import torch
import torch.nn.functional as F


class Discretize:
    """
    Discretization functions for projecting soft embeddings to hard token ids.
    """
    
    # Cache for normalized vocabulary matrices to avoid repeated normalization
    _normalized_vocab_cache: dict[int, torch.Tensor] = {}

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
        
        # Cache normalized vocabulary to avoid repeated normalization
        # Use id() as cache key for this specific tensor
        vocab_id = id(vocab_matrix)
        if vocab_id not in Discretize._normalized_vocab_cache:
            Discretize._normalized_vocab_cache[vocab_id] = F.normalize(vocab_matrix, p=2, dim=-1)
            # Limit cache size to prevent memory issues
            if len(Discretize._normalized_vocab_cache) > 10:
                # Remove oldest entry (first item)
                Discretize._normalized_vocab_cache.pop(next(iter(Discretize._normalized_vocab_cache)))
        
        w = Discretize._normalized_vocab_cache[vocab_id]  # [v, d]

        # cosine sim == dot product for unit vectors
        sims = torch.matmul(q, w.T)  # [b, n, v]
        if forbidden_mask is not None:
            sims = sims.masked_fill(forbidden_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        ids = sims.argmax(dim=-1)  # [b, n]
        return ids
