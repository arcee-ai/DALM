import argparse
import json
import logging
import os
import pickle
import random
from enum import Enum
from typing import Optional

import datasets
import sentencepiece as spm

from dalm.datasets.reading_comprehension_generation.regex_based import RegexBasedReadingComprehension
from dalm.datasets.reading_comprehension_generation.synthetic_based import generate_synthetic_dataset
from dalm.datasets.reading_comprehension_generation.utils import (
    create_domain_tokenizer_from_files,
    question_and_answer_extractor,
)
from dalm.training.generator_only.trainer import train_generator

logger = logging.getLogger(__name__)


class SynthMode(Enum):
    REGEX = "regex"
    LLM = "llm"
    BOTH = "both"


def pipeline(
    model_name: str,
    output_dataset_name: str,
    comprehension_type: SynthMode = SynthMode.BOTH,
    llm_synth_model_name: Optional[str] = None,
    llm_synth_model_context_length: Optional[int] = 4192,
    llm_dataset_output_path: Optional[str] = "llm_dataset",
    general_spm_path: Optional[str] = None,
    domain_spm_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
    chunk: Optional[bool] = False,
    num_train_epochs: Optional[int] = 1,
    split: Optional[str] = "train",
    size_valid_set: Optional[int] = 1000,
    streaming: Optional[bool] = False,
    shuffle_buffer: Optional[int] = 10000,
    seq_length: Optional[int] = 2600,
    num_workers: Optional[int] = 4,
    eval_steps: Optional[int] = 200,
    logging_steps: Optional[int] = 1000,
    per_device_train_batch_size: Optional[int] = 1,
    per_device_eval_batch_size: Optional[int] = 1,
    gradient_accumulation_steps: Optional[int] = 1,
    gradient_checkpointing: Optional[bool] = False,
    group_by_length: Optional[bool] = False,
    packing: Optional[bool] = False,
    lora_alpha: Optional[int] = 16,
    lora_dropout: Optional[float] = 0.05,
    lora_r: Optional[int] = 8,
    learning_rate: Optional[float] = 5e-5,
    lr_scheduler_type: Optional[str] = "linear",
    num_warmup_steps: Optional[int] = 0,
    weight_decay: Optional[float] = 0.0,
    optimizer_type: Optional[str] = "paged_adamw_32bit",
    model_output_dir: Optional[str] = "model_output_dir",
    log_freq: Optional[int] = 100,
    neftune_noise_alpha: Optional[int] = 5,
    log_with: Optional[str] = "wandb",
    generation_state_file: Optional[str] = "generation_state.pkl",
) -> None:
    if comprehension_type in [SynthMode.LLM, SynthMode.BOTH]:
        if not llm_synth_model_name:
            raise ValueError("llm_synth_model_name is required for LLM based generation")
        if not llm_dataset_output_path:
            raise ValueError("llm_dataset_output_path is required for LLM based generation")

    if comprehension_type in [SynthMode.REGEX, SynthMode.BOTH]:
        if not domain_spm_path:
            logger.warning("No domain tokenizer provided. The domain tokenizer will be created from the input files")
            domain_spm = create_domain_tokenizer_from_files(dataset_path)
        else:
            domain_spm = spm.SentencePieceProcessor(model_file=domain_spm_path)

        if not general_spm_path:
            raise ValueError("general_spm_path is required for regex based generation")

        general_spm = spm.SentencePieceProcessor(model_file=general_spm_path)

    in_memory_dataset = []

    # generate regex based reading comprehension dataset
    if comprehension_type in [SynthMode.REGEX, SynthMode.BOTH]:
        # generate regex based reading comprehension dataset
        regex_rc_gen = RegexBasedReadingComprehension(general_spm, domain_spm)

        # NOTE: this is a simple check to see if the dataset is already generated
        in_memory_dataset.extend([{"messages": rc_text} for _, _, rc_text in regex_rc_gen.dataset_generator(dataset_path)])

    generation_state = None
    if generation_state_file:
        if os.path.exists(generation_state_file):
            with open(generation_state_file, "rb") as f:
                generation_state = pickle.load(f)
        else:
            generation_state = {"processed_files": [], "total_files": 0, "files_missed": 0}
            pickle.dump(generation_state, open(generation_state_file, "wb"))

    if not os.path.exists(llm_dataset_output_path):
        os.makedirs(llm_dataset_output_path)

    # NOTE: this operation is time consuming and very expensive
    # Attention has been paid to try to save intermediate steps in case of failure
    # so that the generation can be resumed from the last checkpoint
    if comprehension_type in [SynthMode.LLM, SynthMode.BOTH]:
        llm_rc_dataset_generator = generate_synthetic_dataset(
            model_name=llm_synth_model_name,
            input_directory=dataset_path,
            processed_files=generation_state["processed_files"],
            chunk=chunk,
            context_length=llm_synth_model_context_length,
        )

        # generate llm based reading comprehension dataset
        for index, filename, context, gen_text in llm_rc_dataset_generator:
            qanda = question_and_answer_extractor(gen_text, context)
            if qanda:
                output_file = f"{filename}_{index}.json"
                with open(os.path.join(llm_dataset_output_path, output_file), "w") as o:
                    json.dump(qanda, o)
            else:
                logger.warning(f"No question and answer pairs found for {filename}")
                generation_state["files_missed"] += 1
            generation_state["processed_files"].append(filename)
            generation_state["total_files"] += 1
            pickle.dump(generation_state, open(generation_state_file, "wb"))

        logger.info(" Statistics ")
        logger.info(
            f"Total number of successfully extracted q&a: {generation_state['total_files'] - generation_state['files_missed']}"
        )
        logger.info(f"Total files missed: {generation_state['files_missed']} out of {generation_state['total_files']}")
        logger.info(f"Total files processed: {generation_state['total_files']}")

        for file in os.listdir(llm_dataset_output_path):
            with open(os.path.join(llm_dataset_output_path, file), "r") as f:
                in_memory_dataset.append({"messages": json.load(f)})

    # shuffle in memory dataset
    random.shuffle(in_memory_dataset)

    dataset = datasets.Dataset.from_list(in_memory_dataset)

    dataset.save_to_disk(output_dataset_name)

    train_generator(
        model_name=model_name,
        dataset_name=output_dataset_name,
        num_train_epochs=num_train_epochs,
        split=split,
        size_valid_set=size_valid_set,
        streaming=streaming,
        shuffle_buffer=shuffle_buffer,
        seq_length=seq_length,
        num_workers=num_workers,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        group_by_length=group_by_length,
        packing=packing,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_r=lora_r,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        num_warmup_steps=num_warmup_steps,
        weight_decay=weight_decay,
        optimizer_type=optimizer_type,
        output_dir=model_output_dir,
        log_freq=log_freq,
        neftune_noise_alpha=neftune_noise_alpha,
        log_with=log_with,
        local_dataset=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="name of the model to be trained")
    parser.add_argument(
        "--output_dataset_name",
        type=str,
        required=True,
        help="name of the dataset that the generated data will be saved to",
    )
    parser.add_argument(
        "--comprehension_type",
        type=SynthMode,
        default=SynthMode.BOTH,
        choices=list(SynthMode),
        help="type of comprehension to be generated",
    )
    parser.add_argument(
        "--llm_synth_model_name", type=str, default=None, help="name of the model to be used for LLM based generation"
    )
    parser.add_argument(
        "--llm_synth_model_context_length", type=int, default=4192, help="context length to calulcate the chunk size"
    )
    parser.add_argument(
        "--llm_dataset_output_path",
        type=str,
        default="llm_dataset",
        help="path to save the generated LLM based dataset",
    )
    parser.add_argument(
        "--general_spm_path",
        type=str,
        default=None,
        help="path to the general tokenizer (needed for regex based generation)",
    )
    parser.add_argument(
        "--domain_spm_path",
        type=str,
        default=None,
        help="path to the domain tokenizer (needed for regex based generation)",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="path to the dataset to be used for generation input"
    )
    parser.add_argument("--no_chunk", action="store_true", help="whether to NOT chunk the input files or not")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="number of epochs to train the generator")
    parser.add_argument("--split", type=str, default="train", help="split to be used for training")
    parser.add_argument("--size_valid_set", type=int, default=1000, help="size of the validation set (STREAMING ONLY)")
    parser.add_argument("--streaming", action="store_true", help="whether to use streaming or not")
    parser.add_argument("--shuffle_buffer", type=int, default=10000, help="shuffle buffer size (STREAMING ONLY)")
    parser.add_argument("--seq_length", type=int, default=2600, help="sequence length to be used for training")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="number of workers to be used for data loading during training"
    )
    parser.add_argument("--eval_steps", type=int, default=200, help="number of steps to evaluate the model")
    parser.add_argument("--logging_steps", type=int, default=1000, help="number of steps to log the model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="batch size to be used for training")
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=1, help="batch size to be used for evaluation"
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument(
        "--no_gradient_checkpointing", action="store_true", help="whether to disable gradient checkpointing"
    )
    parser.add_argument("--group_by_length", action="store_true", help="whether to group the dataset by length or not")
    parser.add_argument("--no_packing", action="store_true", help="whether to disable packing or not")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="lora dropout")
    parser.add_argument("--lora_r", type=int, default=8, help="lora r")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer_type", type=str, default="paged_adamw_32bit")
    parser.add_argument("--model_output_dir", type=str, default="model_output_dir")
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--neftune_noise_alpha", type=int, default=5)
    parser.add_argument(
        "--log_with",
        type=str,
        default="wandb",
        help="tracker backend to be used",
    )
    parser.add_argument(
        "--generation_state_file", type=str, default="generation_state.pkl", help="file to save the generation state to"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline(
        model_name=args.model_name,
        output_dataset_name=args.output_dataset_name,
        comprehension_type=args.comprehension_type,
        llm_synth_model_name=args.llm_synth_model_name,
        llm_synth_model_context_length=args.llm_synth_model_context_length,
        llm_dataset_output_path=args.llm_dataset_output_path,
        general_spm_path=args.general_spm_path,
        domain_spm_path=args.domain_spm_path,
        dataset_path=args.dataset_path,
        chunk=not args.no_chunk,
        num_train_epochs=args.num_train_epochs,
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
        gradient_checkpointing=not args.no_gradient_checkpointing,
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
        model_output_dir=args.model_output_dir,
        log_freq=args.log_freq,
        neftune_noise_alpha=args.neftune_noise_alpha,
        log_with=args.log_with,
        generation_state_file=args.generation_state_file,
    )


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()
