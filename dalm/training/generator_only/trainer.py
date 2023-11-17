import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

import argparse


# @dataclass
# class ScriptArguments:
#    model_name: Optional[str] = field(default="HuggingFaceH4/zephyr-7b-alpha", metadata={"help": "the model name"})
#    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
#    dataset_name: Optional[str] = field(
#        default="arcee-ai/azure-reading-comprehension-dataset", metadata={"help": "the dataset name"}
#    )
#    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
#    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
#    streaming: Optional[bool] = field(default=False, metadata={"help": "whether to stream the dataset"})
#    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
#    seq_length: Optional[int] = field(default=2600, metadata={"help": "the sequence length"})
#    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
#
#    eval_steps: Optional[int] = field(default=200, metadata={"help": "the evaluation frequency"})
#    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
#    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device train batch size"})
#    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
#    gradient_accumulation_steps: Optional[int] = field(default=32, metadata={"help": "the gradient accumulation steps"})
#    gradient_checkpointing: Optional[bool] = field(
#        default=True, metadata={"help": "whether to use gradient checkpointing"}
#    )
#    group_by_length: Optional[bool] = field(default=False, metadata={"help": "whether to group by length"})
#    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})
#
#    lora_alpha: Optional[float] = field(default=512, metadata={"help": "the lora alpha parameter"})
#    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
#    lora_r: Optional[int] = field(default=256, metadata={"help": "the lora r parameter"})
#
#    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
#    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
#    num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
#    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
#    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
#
#    output_dir: Optional[str] = field(default="./results2", metadata={"help": "the output directory"})
#    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})


def create_datasets(
    dataset_name: str,
    split: str,
    size_valid_set: Optional[int],
    streaming: bool,
    shuffle_buffer: int,
    seq_length: int,
    num_workers: int,
    tokenizer: AutoTokenizer,
    formatting_func: callable,
):
    dataset = load_dataset(
        dataset_name,
        split=split,
        num_proc=num_workers if not streaming else None,
        streaming=streaming,
    )
    if streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(size_valid_set)
        train_data = dataset.skip(size_valid_set)
        train_data = train_data.shuffle(buffer_size=shuffle_buffer, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=0.05, seed=None)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer, formatting_func)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=formatting_func,
        infinite=True,
        seq_length=seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=formatting_func,
        infinite=False,
        seq_length=seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


def chars_token_ratio(dataset, tokenizer, formatting_func, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = formatting_func(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


# def print_trainable_parameters(model):
#    """
#    Prints the number of trainable parameters in the model.
#    """
#    trainable_params = 0
#    all_param = 0
#    for _, param in model.named_parameters():
#        all_param += param.numel()
#        if param.requires_grad:
#            trainable_params += param.numel()
#    print(
#        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
#    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="HuggingFaceH4/zephyr-7b-alpha", help="the model name")
    parser.add_argument("--log_with", type=str, default="wandb", help="use 'wandb' to log with wandb")
    parser.add_argument(
        "--dataset_name", type=str, default="arcee-ai/azure-reading-comprehension-dataset", help="the dataset name"
    )
    parser.add_argument("--split", type=str, default="train", help="the split to use")
    parser.add_argument("--size_valid_set", type=int, default=4000, help="the size of the validation set")
    parser.add_argument("--streaming", type=bool, default=False, help="whether to stream the dataset")
    parser.add_argument("--shuffle_buffer", type=int, default=5000, help="the shuffle buffer size")
    parser.add_argument("--seq_length", type=int, default=2600, help="the sequence length")
    parser.add_argument("--num_workers", type=int, default=4, help="the number of workers")

    parser.add_argument("--eval_steps", type=int, default=200, help="the evaluation frequency")
    parser.add_argument("--logging_steps", type=int, default=10, help="the logging frequency")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="the per device train batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="the per device eval batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="the gradient accumulation steps")
    parser.add_argument(
        "--gradient_checkpointing", type=bool, default=True, help="whether to use gradient checkpointing"
    )
    parser.add_argument("--group_by_length", type=bool, default=False, help="whether to group by length")
    parser.add_argument("--packing", type=bool, default=True, help="whether to use packing for SFTTrainer")

    parser.add_argument("--lora_alpha", type=float, default=512, help="the lora alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="the lora dropout parameter")
    parser.add_argument("--lora_r", type=int, default=256, help="the lora r parameter")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="the learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="the lr scheduler type")
    parser.add_argument("--num_warmup_steps", type=int, default=100, help="the number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="the weight decay")
    parser.add_argument("--optimizer_type", type=str, default="paged_adamw_32bit", help="the optimizer type")

    parser.add_argument("--output_dir", type=str, default="./results2", help="the output directory")
    parser.add_argument("--neftune_noise_alpha", type=int, default=5, help="the noise alpha for neftune")
    parser.add_argument("--log_freq", type=int, default=1, help="the logging frequency")
    return parser.parse_args()


def train_generator(
    model_name,
    dataset_name,
    split,
    size_valid_set,
    streaming,
    shuffle_buffer,
    seq_length,
    num_workers,
    eval_steps,
    logging_steps,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    gradient_accumulation_steps,
    gradient_checkpointing,
    group_by_length,
    packing,
    lora_alpha,
    lora_dropout,
    lora_r,
    learning_rate,
    lr_scheduler_type,
    num_warmup_steps,
    weight_decay,
    optimizer_type,
    output_dir,
    log_freq,
    neftune_noise_alpha=5,
    log_with="wandb",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": Accelerator().local_process_index},
            trust_remote_code=True,
            use_auth_token=True,
        ),
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
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        num_train_epochs=3,
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
        run_name="generator_tuning",
    )

    def prepare_sample_text(example):
        """Prepare the text from a sample of the dataset."""
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return text

    train_dataset, eval_dataset = create_datasets(
        dataset_name,
        split,
        size_valid_set,
        streaming,
        shuffle_buffer,
        seq_length,
        num_workers,
        tokenizer,
        prepare_sample_text,
    )

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=packing,
        max_seq_length=seq_length,
        tokenizer=tokenizer,
        args=training_args,
        neftune_noise_alpha=neftune_noise_alpha,
    )

    trainer.train()
    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)


def main():
    args = parse_args()
    train_generator(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        split=args.split,
        size_valid_set=args.size_valid_set,
        streaming=args.streaming,
        shuffle_buffer=args.shuffle_buffer,
        seq_length=args.seq_length,
        num_workers=args.num_workers,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        group_by_length=args.group_by_length,
        packing=args.packing,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_r=args.lora_r,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        num_warmup_steps=args.num_warmup_steps,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer_type,
        output_dir=args.output_dir,
        log_freq=args.log_freq,
        log_with=args.log_with,
        neftune_noise_alpha=args.neftune_noise_alpha,
    )


if __name__ == "__main__":
    main()
