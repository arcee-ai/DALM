from typing import List

from typing import Any, Dict
from datasets.formatting.formatting import LazyBatch

import torch
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedTokenizer, PreTrainedTokenizerFast


def save_model_hook(models: List[AutoModel], weights: List, output_dir: str) -> None:
    for i, model in enumerate(models):
        model.save_pretrained(output_dir, state_dict=weights[i])
        # make sure to pop weight so that corresponding model is not saved again
        weights.pop()


def load_model_hook(models: List[AutoModel], input_dir: str) -> None:
    while len(models) > 0:
        model = models.pop()
        # pop models so that they are not loaded again
        if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
            model.load_adapter(input_dir, model.active_adapter, is_trainable=True)


def get_cosine_sim(
    query_embs: torch.FloatTensor, passage_embs: torch.FloatTensor, logit_scale: torch.FloatTensor
) -> torch.Tensor:
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


def preprocess_dataset(
    examples: LazyBatch,
    retriever_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    generator_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    dataset_query_col_name: str,
    dataset_passage_col_name: str,
    dataset_answer_col_name: str,
) -> Dict[str, Any]:
    queries = examples[dataset_query_col_name]
    passages = examples[dataset_passage_col_name]
    answers = examples[dataset_answer_col_name]

    # Tokenization for the retriever
    retriever_query_tokens = retriever_tokenizer(queries, padding="max_length", max_length=128, truncation=True)
    retriever_passage_tokens = retriever_tokenizer(passages, padding="max_length", max_length=128, truncation=True)

    # Tokenize for causal model
    # Here, we need to combine the query, passage, and the answer as the input, and the answer as the output
    casual_input_text = [
        f"#query# {query} #passage# {passage} #answer# {answer}"
        for passage, query, answer in zip(passages, queries, answers, strict=True)
    ]
    causal_input_tokens = generator_tokenizer(casual_input_text, padding="max_length", max_length=128, truncation=True)

    query_passage_text = [
        f"#query# {query} #passage# {passage} #answer# "
        for passage, query, answer in zip(passages, queries, answers, strict=True)
    ]

    query_passage_lengths = []

    query_passage_tokens = generator_tokenizer(query_passage_text)

    for single_query_passage in query_passage_tokens["input_ids"]:
        query_passage_lengths.append(len(single_query_passage))

    pre_batch = {}

    # for the retriever in-batch negats
    for k, v in retriever_query_tokens.items():
        pre_batch[f"retriever_query_{k}"] = v
    for k, v in retriever_passage_tokens.items():
        pre_batch[f"retriever_passage_{k}"] = v

    # for the generator
    for k, v in causal_input_tokens.items():
        pre_batch[f"generator_input_{k}"] = v

    pre_batch["query_passage_input_len"] = query_passage_lengths

    return pre_batch