# TODO: actually implement properly

import torch


# from PEZ paper
# here the similarity is based on dot-product
# funcions are imported from sentence_transformers
def nn_project(curr_embeds, embedding_layer, print_hits=False):
    with torch.no_grad():
        bsz, seq_len, emb_dim = curr_embeds.shape

        # Using the sentence transformers semantic search which is
        # a dot product exact kNN search between a set of
        # query vectors and a corpus of vectors
        curr_embeds = curr_embeds.reshape((-1, emb_dim))
        curr_embeds = normalize_embeddings(curr_embeds)  # queries

        embedding_matrix = embedding_layer.weight
        embedding_matrix = normalize_embeddings(embedding_matrix)

        hits = semantic_search(
            curr_embeds,
            embedding_matrix,
            query_chunk_size=curr_embeds.shape[0],
            top_k=1,
            score_function=dot_score,
        )

        if print_hits:
            all_hits = []
            for hit in hits:
                all_hits.append(hit[0]["score"])
            print(f"mean hits:{mean(all_hits)}")

        nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device)
        nn_indices = nn_indices.reshape((bsz, seq_len))

        projected_embeds = embedding_layer(nn_indices)

    return projected_embeds, nn_indices


# from RR paper
def find_closest_embeddings(
    embeddings_adv: torch.Tensor,
    embed_weights: torch.Tensor,
    device,
    allow_non_ascii=True,
    non_ascii_toks=None,
):
    def normalize(v: torch.Tensor) -> torch.Tensor:
        return v / torch.norm(v, p=2)

    embeddings_adv = normalize(embeddings_adv)
    embed_weights = normalize(embed_weights)
    distances = torch.cdist(embeddings_adv, embed_weights, p=2)
    if not allow_non_ascii:
        distances[0][:, non_ascii_toks.to(device)] = float("inf")
    closest_distances, closest_indices = torch.min(distances, dim=-1)
    closest_embeddings = embed_weights[closest_indices]
    return closest_distances, closest_indices, closest_embeddings
