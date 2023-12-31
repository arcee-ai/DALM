import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModel

from dalm.models.rag_e2e_base_model import AutoModelForRagE2E
from dalm.models.retriever_only_base_model import AutoModelForSentenceEmbedding


def extract_sub_state_dict(full_state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix) :]: v for k, v in full_state_dict.items() if k.startswith(prefix)}


def save_model_hook(models: List[AutoModel], weights: List, output_dir: str) -> None:
    for i, model in enumerate(models):
        if isinstance(model, AutoModelForSentenceEmbedding):
            formatted_weights = extract_sub_state_dict(weights[i], "model.")
            model.model.save_pretrained(output_dir, state_dict=formatted_weights)
        elif isinstance(model, AutoModelForRagE2E):
            combined_weights = weights[i]
            generator_weights = extract_sub_state_dict(combined_weights, "generator_model.")
            retriever_weights = extract_sub_state_dict(combined_weights, "retriever_model.")

            model.generator_model.save_pretrained(os.path.join(output_dir, "generator"), state_dict=generator_weights)
            model.retriever_model.save_pretrained(os.path.join(output_dir, "retriever"), state_dict=retriever_weights)
        else:
            raise NotImplementedError
        # make sure to pop weight so that corresponding model is not saved again
        weights.pop()


def load_model_hook(models: List[AutoModel], input_dir: str) -> None:
    while len(models) > 0:
        model = models.pop()
        # pop models so that they are not loaded again
        if isinstance(model, AutoModelForRagE2E):
            generator_path = os.path.join(input_dir, "generator")
            adapter_model_path = os.path.join(generator_path, "adapter_model.bin")

            if (
                hasattr(model.generator_model, "active_adapter")
                and hasattr(model.generator_model, "load_adapter")
                and os.path.exists(adapter_model_path)
            ):
                model.generator_model.load_adapter(generator_path, model.generator_model.active_adapter)
            else:
                model.generator_model = AutoModel.from_pretrained(generator_path)

            retriever_path = os.path.join(input_dir, "retriever")
            adapter_model_path = os.path.join(retriever_path, "adapter_model.bin")

            if (
                hasattr(model.retriever_model, "active_adapter")
                and hasattr(model.retriever_model, "load_adapter")
                and os.path.exists(adapter_model_path)
            ):
                model.retriever_model.load_adapter(retriever_path, model.retriever_model.active_adapter)
            else:
                model.retriever_model = AutoModel.from_pretrained(retriever_path)
        elif isinstance(model, AutoModelForSentenceEmbedding):
            adapter_model_path = os.path.join(input_dir, "adapter_model.bin")
            if (
                hasattr(model.model, "active_adapter")
                and hasattr(model, "load_adapter")
                and os.path.exists(adapter_model_path)
            ):
                model.model.load_adapter(input_dir, model.model.active_adapter)
            else:
                model.model = AutoModel.from_pretrained(input_dir)
        else:
            model.load_adapter(input_dir, model.active_adapter, is_trainable=True)


def get_cosine_sim(query_embs: torch.FloatTensor, passage_embs: torch.FloatTensor, logit_scale: int) -> torch.Tensor:
    return torch.matmul(query_embs, passage_embs.t()) * logit_scale


def get_nt_xent_loss(sim_scores: torch.Tensor) -> torch.Tensor:
    # Took from: https://github.com/UKPLab/sentence-transformers/blob/master
    # /sentence_transformers/losses/MultipleNegativesRankingLoss.py
    """
    Compute the cross_entropy given the square similarity matrix.
    This indicates that `a_i` and `b_j` have high similarity
    when `i==j` and low similarity when `i!=j`.
    """
    return torch.nn.functional.cross_entropy(sim_scores, torch.arange(len(sim_scores), device=sim_scores.device))


def get_nll(log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    ll = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(-1)
    return -ll


def marginalize_log_probs(
    logprobs_logits: torch.FloatTensor, doc_logprobs: torch.FloatTensor, query_token_length: torch.IntTensor
) -> torch.Tensor:
    # we let the model to predict the next word for the query as is
    query_passage_log_prob = logprobs_logits[: query_token_length - 1, :]

    # we marginalize the next word prediction of the passa based on the scores
    answer_log_prob = logprobs_logits[query_token_length - 1 :, :]

    marginalized_prob_sum = answer_log_prob + doc_logprobs

    # get all the log probs
    all_log_probs = torch.cat([query_passage_log_prob, marginalized_prob_sum], dim=0)

    return all_log_probs


def compute_marginalized_loss_from_logits(
    logits: torch.LongTensor,
    input_tensors: torch.Tensor,
    attention_mask: torch.Tensor,
    scores: torch.Tensor,
    query_token_length: torch.Tensor,
) -> torch.Tensor:
    # do not compute for the final token
    logprobs_logits = F.log_softmax(logits[:, :-1, :], dim=2).view(logits.shape[0], -1, logits.size(-1))

    # here I am assuming that we always take the positive sample as the correct one
    doc_logprobs = torch.log_softmax(scores, dim=1).diag().unsqueeze(-1).unsqueeze(-1)
    marginalized_next_word_prob_list = []

    for sample_logprobs_logits, sample_doc_logprobs, sample_token_length in zip(
        logprobs_logits, doc_logprobs, query_token_length, strict=True
    ):
        marginalized_log_probs = marginalize_log_probs(sample_logprobs_logits, sample_doc_logprobs, sample_token_length)
        marginalized_next_word_prob_list.append(marginalized_log_probs)

    marginalized_log_probs = torch.stack(marginalized_next_word_prob_list)
    loss = get_nll(marginalized_log_probs, input_tensors[:, 1:])
    loss_tensor = loss * attention_mask[:, 1:]
    overall_average_loss = loss_tensor.sum() / attention_mask[:, 1:].sum()

    return overall_average_loss
