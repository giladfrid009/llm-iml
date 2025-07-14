import torch
from sentence_transformers.util import semantic_search, dot_score, normalize_embeddings
from torch.nn import CrossEntropyLoss

from src.sample_attacks.harmbench.model_utils import get_template
from src.sample_attacks.harmbench.baseline import IndivHarmBenchAttack
from src.adv_model import AdvModel


# ============================== PEZ CLASS DEFINITION ============================== #
class PEZ(IndivHarmBenchAttack):
    def __init__(
        self,
        adv_model: AdvModel,
        num_optim_tokens=20,
        num_steps=100,
        lr=1e-3,
        verbose: bool = True,
    ):
        """
        :param target_model: a dictionary specifying the target model (kwargs to load_model_and_tokenizer)
        :param num_optim_tokens: the number of tokens to optimize
        :param num_steps: the number of optimization steps
        :param lr: the learning rate for soft embeddings
        """
        super().__init__(adv_model, verbose)
        self.num_optim_tokens = num_optim_tokens
        self.num_steps = num_steps
        self.lr = lr

        template = get_template(self.tokenizer)
        self.template = template
        self.before_tc, self.after_tc = template.split("{instruction}")

    def generate_test_cases_single_behavior(
        self, behavior: str, target: str, init_embeds: torch.Tensor | None = None
    ) -> str:
        """
        Generates test cases for a single behavior

        :param behavior: a dictionary specifying the behavior to generate test cases for
        :param num_generate: the number of test cases to generate in parallel
        :param verbose: whether to print progress
        :return: a list of test case and a list of logs
        """
        # PEZ hyperparams
        num_generate = 1
        num_optim_tokens = self.num_optim_tokens
        num_steps = self.num_steps
        lr = self.lr

        # Vars
        model = self.model
        tokenizer = self.tokenizer
        device = model.device
        before_tc = self.before_tc
        after_tc = self.after_tc
        embed_layer: torch.nn.Embedding = self.model.get_input_embeddings()
        project_soft_embeds.embed_layer = embed_layer  # have to set this here to allow using different models

        # ========== Init Cache Embeds ========
        cache_input_ids = tokenizer([before_tc], padding=False)["input_ids"]  # some tokenizer have <s> for before_tc
        cache_input_ids += tokenizer([behavior, after_tc, target], padding=False, add_special_tokens=False)["input_ids"]
        cache_input_ids = [
            torch.tensor(input_ids, device=device).unsqueeze(0) for input_ids in cache_input_ids
        ]  # make tensor separately because can't return_tensors='pt' in tokenizer
        before_ids, behavior_ids, after_ids, target_ids = cache_input_ids
        before_embeds, behavior_embeds, after_embeds, target_embeds = [
            embed_layer(input_ids) for input_ids in cache_input_ids
        ]

        # ========== setup optim_embeds, optimizer, and scheduler ========== #
        optim_embeds = torch.randint(embed_layer.weight.shape[0], (num_generate, num_optim_tokens), device=device)
        optim_embeds = embed_layer(optim_embeds).float()  # convert to float to avoid NaN with Adam optimizer
        optim_embeds = torch.nn.Parameter(optim_embeds)
        optim_embeds.requires_grad_()
        optimizer = torch.optim.Adam([optim_embeds], lr=lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)

        # ========== run optimization ========== #
        for i in range(num_steps):
            # ========== compute logits with concatenated optim embeds and target text ========== #
            optim_embeds_projected = project_soft_embeds.apply(optim_embeds.to(model.dtype))

            input_embeds = torch.cat(
                [
                    before_embeds.repeat(num_generate, 1, 1),
                    behavior_embeds.repeat(num_generate, 1, 1),
                    optim_embeds_projected,
                    after_embeds.repeat(num_generate, 1, 1),
                    target_embeds.repeat(num_generate, 1, 1),
                ],
                dim=1,
            )

            outputs = self.model(inputs_embeds=input_embeds)
            logits: torch.Tensor = outputs.logits

            # ========== compute loss ========== #
            # Shift so that tokens < n predict n
            tmp = input_embeds.shape[1] - target_embeds.shape[1]
            shift_logits = logits[..., tmp - 1 : -1, :].contiguous()
            shift_labels = target_ids.repeat(num_generate, 1)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct.forward(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(num_generate, -1).mean(dim=1)

            # ========== update optim_embeds ========== #
            optimizer.zero_grad()
            loss.mean().backward(inputs=[optim_embeds])
            optimizer.step()
            scheduler.step()

            # ========== retrieve optim_tokens and test_cases========== #
            with torch.no_grad():
                # ========== detokenize and print the optimized prompt ========== #
                _, nn_indices = nn_project(optim_embeds.to(model.dtype), embed_layer)

            nn_indices = torch.cat([behavior_ids.repeat(num_generate, 1), nn_indices], dim=1)
            test_cases = tokenizer.batch_decode(nn_indices, skip_special_tokens=True)

        return test_cases[0]


# ============================== UTILS FOR PEZ ============================== #
def nn_project(curr_embeds: torch.Tensor, embedding_layer: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, emb_dim = curr_embeds.shape

    # Using the sentence transformers semantic search which is
    # a dot product exact kNN search between a set of
    # query vectors and a corpus of vectors
    curr_embeds = curr_embeds.reshape((-1, emb_dim))
    curr_embeds = normalize_embeddings(curr_embeds)  # queries

    embedding_matrix: torch.Tensor = embedding_layer.weight
    embedding_matrix = normalize_embeddings(embedding_matrix)  # corpus

    hits = semantic_search(
        curr_embeds, embedding_matrix, query_chunk_size=curr_embeds.shape[0], top_k=3, score_function=dot_score
    )

    nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device="cuda")
    projected_embeds = embedding_layer(nn_indices)
    projected_embeds = projected_embeds.reshape((batch_size, seq_len, emb_dim))
    nn_indices = nn_indices.reshape((batch_size, seq_len))

    return projected_embeds, nn_indices


class project_soft_embeds(torch.autograd.Function):
    """
    This is a PyTorch layer that projects the soft embeddings to the nearest
    hard embedding in the forward pass and passes the gradient through in the
    backward pass. This is a straight-through estimator.
    """

    embed_layer = None  # class variable to store the embedding layer

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        projected_embeds, nn_indices = nn_project(input, project_soft_embeds.embed_layer)  # for Pythia
        return projected_embeds

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        (input,) = ctx.saved_tensors
        return grad_output  # straight-through estimator
