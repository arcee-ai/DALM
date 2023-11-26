import argparse

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import CollateFnWithTokenization, InferenceDataset
from src.utils.io import dump_jsonlines, load_jsonlines


@torch.inference_mode()
def main(args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.half()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    dataset = InferenceDataset(
        load_jsonlines(args.data_filepath),
        content_name=args.prompt_column_name,
        reverse=args.reverse,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=CollateFnWithTokenization(tokenizer),
        shuffle=False,
    )

    model, data_loader = accelerator.prepare(model, data_loader)

    results = []

    for batch in tqdm(data_loader):
        output_ids = accelerator.unwrap_model(model).generate(
            **batch,
            do_sample=True if args.temperature > 1e-5 else False,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )

        output_ids = accelerator.pad_across_processes(
            output_ids, dim=1, pad_index=tokenizer.pad_token_id
        )
        input_ids = accelerator.gather(batch["input_ids"]).cpu().numpy()
        output_ids = accelerator.gather(output_ids).cpu().numpy()

        decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for decoded_input, decoded_pred in zip(decoded_inputs, decoded_preds):
            results.append(
                {
                    "prompt": decoded_input,
                    "response": decoded_pred,
                }
            )
            results.append(decoded_pred)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        dump_jsonlines(results, args.save_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--data_filepath", type=str)
    parser.add_argument("--save_filepath", type=str)
    parser.add_argument("--prompt_column_name", type=str, default="instruction")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    args = parser.parse_args()

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    main(args)
