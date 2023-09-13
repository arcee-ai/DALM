from typing import List

import torch
from transformers import BartForConditionalGeneration, BartTokenizerFast


def generate_question(
    passages: List[str],
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizerFast,
) -> List[str]:
    """Given a passage, create a question that would be answered by it"""
    inputs = tokenizer(
        passages,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.autocast("cuda"), torch.no_grad():
        outputs = model.generate(**inputs.to("cuda"), max_length=50, num_beams=1, early_stopping=True).cpu()
    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
