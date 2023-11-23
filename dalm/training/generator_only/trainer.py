import argparse
import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset, load_from_disk
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from trl import SFTTrainer  # type: ignore[import]

logger = logging.getLogger(__name__)


def create_datasets(
    dataset_name: str,
    split: str,
    validation_split: Optional[float],
    size_valid_set: Optional[int],
    streaming: bool,
    shuffle_buffer: Optional[int],
    num_workers: int,
    local_dataset: bool = False,
) -> Tuple[Dataset, Dataset]:
    if local_dataset:
        dataset = load_from_disk(
            dataset_name,
        )
    else:
        dataset = load_dataset(
            dataset_name,
            split=split,
            num_proc=num_workers if not streaming else None,
            streaming=streaming,
        )
    if streaming:
        logging.info("Loading the dataset in streaming mode")
        if not (shuffle_buffer and size_valid_set):
            raise ValueError("size_valid_set must be set when streaming is enabled")
        valid_data = dataset.take(size_valid_set)
        train_data = dataset.skip(size_valid_set)
        train_data = train_data.shuffle(buffer_size=shuffle_buffer, seed=None)
    else:
        if not validation_split:
            raise ValueError("validation_split must be set when streaming is disabled")
        dataset = dataset.train_test_split(test_size=validation_split, seed=None)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        logging.info(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    return train_data, valid_data


def chars_token_ratio(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    formatting_func: Callable[[Dict[str, Any]], str],
    sample_size: int = 400,
) -> float:
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(sample_size), iter(dataset), strict=False), total=sample_size):
        text = formatting_func(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="HuggingFaceH4/zephyr-7b-beta", help="the model name")
    parser.add_argument("--log_with", type=str, default="wandb", help="tracker name (wandb, mlflow, ..etc)")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help=(
            "The dataset name corresponding to one sitting on huggingface or a local one"
            "If local, be sure to set the local_dataset flag"
        ),
    )
    parser.add_argument("--local_dataset", action="store_true", help="whether to use a local dataset")
    parser.add_argument("--split", type=str, default="train", help="the split to use")
    parser.add_argument(
        "--size_valid_set", type=int, default=4000, help="the size of the validation set (when streaming is enabled)"
    )
    parser.add_argument("--validation_split", type=float, default=0.05, help="the validation split percentage")
    parser.add_argument("--stream", action="store_true", help="whether to stream the dataset")
    parser.add_argument("--shuffle_buffer", type=int, default=5000, help="the shuffle buffer size")
    parser.add_argument("--seq_length", type=int, default=2600, help="the sequence length")
    parser.add_argument("--num_workers", type=int, default=4, help="the number of workers")

    parser.add_argument("--eval_steps", type=int, default=200, help="the evaluation frequency")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="the number of training epochs")
    parser.add_argument("--logging_steps", type=int, default=10, help="the logging frequency")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="the per device train batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="the per device eval batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="the gradient accumulation steps")
    parser.add_argument(
        "--gradient_checkpointing", type=bool, default=True, help="whether to use gradient checkpointing"
    )
    parser.add_argument("--group_by_length", type=bool, default=False, help="whether to group by length")
    parser.add_argument("--no-packing", action="store_true", help="whether to not pack the sequences")

    parser.add_argument("--lora_alpha", type=float, default=512, help="the lora alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="the lora dropout parameter")
    parser.add_argument("--lora_r", type=int, default=256, help="the lora r parameter")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="the learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="the lr scheduler type")
    parser.add_argument("--num_warmup_steps", type=int, default=100, help="the number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay")
    parser.add_argument("--optimizer_type", type=str, default="paged_adamw_32bit", help="the optimizer type")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generator_finetuned_model",
        help="the output directory where the model will be saved",
    )
    parser.add_argument("--neftune_noise_alpha", type=int, default=5, help="the noise alpha for neftune")
    parser.add_argument("--run_name", type=str, default="generator_finetuning", help="the tracker run name")
    return parser.parse_args()


def train_generator(
    model_name: str,
    dataset_name: str,
    local_dataset: bool,
    log_with: str,
    run_name: str,
    output_dir: str,
    size_valid_set: Optional[int],
    validation_split: Optional[float],
    shuffle_buffer: Optional[int],
    num_train_epochs: int = 1,
    split: str = "train",
    streaming: bool = False,
    seq_length: int = 2600,
    num_workers: int = 4,
    eval_steps: int = 200,
    logging_steps: int = 10,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    group_by_length: bool = False,
    packing: bool = True,
    lora_alpha: int = 512,
    lora_dropout: float = 0.05,
    lora_r: int = 256,
    learning_rate: float = 1e-4,
    lr_scheduler_type: str = "cosine",
    num_warmup_steps: int = 100,
    weight_decay: float = 0.05,
    optimizer_type: str = "paged_adamw_32bit",
    neftune_noise_alpha: int = 5,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": Accelerator().local_process_index},
        trust_remote_code=True,
    )

    base_model.config.use_cache = False

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        num_train_epochs=num_train_epochs,
        report_to=log_with,
        save_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=num_warmup_steps,
        optim=optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=run_name,
        weight_decay=weight_decay,
        neftune_noise_alpha=neftune_noise_alpha,
    )

    def prepare_sample_text(example: Dict[str, Any]) -> str:
        """Prepare the text from a sample of the dataset."""
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return text

    train_dataset, eval_dataset = create_datasets(
        dataset_name=dataset_name,
        split=split,
        validation_split=validation_split,
        size_valid_set=size_valid_set,
        streaming=streaming,
        shuffle_buffer=shuffle_buffer,
        num_workers=num_workers,
        local_dataset=local_dataset,
    )

    chars_per_token = chars_token_ratio(train_dataset, tokenizer, prepare_sample_text)
    logging.info(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=packing,
        max_seq_length=seq_length,
        tokenizer=tokenizer,
        args=training_args,
        chars_per_token=chars_per_token,
        formatting_func=prepare_sample_text,
    )

    trainer.train()
    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)


def main() -> None:
    args = parse_args()
    train_generator(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        local_dataset=args.local_dataset,
        split=args.split,
        size_valid_set=args.size_valid_set,
        validation_split=args.validation_split,
        streaming=args.stream,
        shuffle_buffer=args.shuffle_buffer,
        seq_length=args.seq_length,
        num_workers=args.num_workers,
        num_train_epochs=args.num_train_epochs,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        group_by_length=args.group_by_length,
        packing=not args.no_packing,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_r=args.lora_r,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        num_warmup_steps=args.num_warmup_steps,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer_type,
        output_dir=args.output_dir,
        log_with=args.log_with,
        neftune_noise_alpha=args.neftune_noise_alpha,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
