import torch
import torch.nn.functional as F
from typing import Dict, Any, Union

def save_model_hook(models, weights, output_dir):
    for i, model in enumerate(models):
        model.save_pretrained(output_dir, state_dict=weights[i])
        # make sure to pop weight so that corresponding model is not saved again
        weights.pop()


def load_model_hook(models, input_dir):
    while len(models) > 0:
        model = models.pop()
        # pop models so that they are not loaded again
        if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
            model.load_adapter(input_dir, model.active_adapter, is_trainable=True)


def get_cosine_sim(query_embs, passage_embs, logit_scale):
    return torch.matmul(query_embs, passage_embs.t()) * logit_scale


def get_nt_xent_loss(sim_scores):
    # Took from: https://github.com/UKPLab/sentence-transformers/blob/master
    # /sentence_transformers/losses/MultipleNegativesRankingLoss.py
    """
    Compute the cross_entropy given the square similarity matrix.
    This indicates that `a_i` and `b_j` have high similarity
    when `i==j` and low similarity when `i!=j`.
    """
    return torch.nn.functional.cross_entropy(
        sim_scores, torch.arange(len(sim_scores), device=sim_scores.device)
    )

def get_nll(log_probs, labels):
    ll = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(-1)
    return -ll


def marginalize_log_probs(logprobs_logits, doc_logprobs, query_token_length):
    
    
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
    scores: torch.FloatTensor,
    query_token_length: Union[int, torch.Tensor],
) -> Dict[str, Any]:
    

    # do not compute for the final token
    logprobs_logits = F.log_softmax(logits[:, :-1, :], dim=2).view(
        logits.shape[0], -1, logits.size(-1)
    )
    
    # here I am assuming that we always take the positive sample as the correct one   
    doc_logprobs = torch.log_softmax(scores, dim=1).diag().unsqueeze(-1).unsqueeze(-1)
    
     
    marginalized_next_word_prob_list = []
    
    for sample_logprobs_logits, sample_doc_logprobs, sample_doc_logprobs in zip (logprobs_logits, doc_logprobs, query_token_length):
        
        marginalized_log_probs = marginalize_log_probs(
            sample_logprobs_logits, 
            sample_doc_logprobs, 
            sample_doc_logprobs
        )
                
        marginalized_next_word_prob_list.append(marginalized_log_probs)
        
    marginalized_log_probs = torch.stack(marginalized_next_word_prob_list)
    
    loss = get_nll(marginalized_log_probs, input_tensors[:, 1:])
    
    
    loss_tensor =  loss * attention_mask[:, 1:]
    
    # Sum the losses for each example (row)
    sum_loss_per_example = torch.sum(loss_tensor, dim=1)

    # Count the number of non-zero elements in each example
    non_zero_counts = torch.count_nonzero(loss_tensor, dim=1)
    
    # Calculate the average loss for each example, considering only non-zero elements
    average_loss_per_example = sum_loss_per_example / non_zero_counts
 
    # Calculate the overall average over the batch
    overall_average_loss = torch.mean(average_loss_per_example)


    return overall_average_loss
