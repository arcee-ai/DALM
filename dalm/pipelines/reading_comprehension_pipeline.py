import argparse
import json
import logging
import os
import pickle
import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import datasets
import sentencepiece as spm  # type: ignore[import-untyped]

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


@dataclass
class LLMKwargs:
    model_name: str
    context_length: Optional[int]
    dataset_output_path: str
    chunk: bool

    def __post_init__(self) -> None:
        if self.chunk and not self.context_length:
            raise ValueError("context_length is required for chunking")


@dataclass
class SynthKwargs:
    general_spm_path: str
    domain_spm_path: Optional[str]


def pipeline(
    model_name: str,
    output_dataset_name: str,
    input: str,
    model_output_dir: str,
    generation_state_file: str,
    llm_kwargs: Optional[LLMKwargs],
    synth_kwargs: Optional[SynthKwargs],
    csv_column: Optional[str],
    size_valid_set: Optional[int],
    comprehension_type: SynthMode,
    shuffle_buffer: Optional[int],
    num_train_epochs: int = 1,
    split: str = "train",
    streaming: bool = False,
    seq_length: int = 2600,
    num_workers: int = 4,
    eval_steps: int = 200,
    logging_steps: int = 1000,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    group_by_length: bool = False,
    packing: bool = True,
    lora_alpha: int = 512,
    lora_dropout: float = 0.05,
    lora_r: int = 256,
    learning_rate: float = 5e-5,
    lr_scheduler_type: str = "cosine",
    num_warmup_steps: int = 0,
    weight_decay: float = 0.0,
    optimizer_type: str = "paged_adamw_32bit",
    neftune_noise_alpha: int = 5,
    log_with: str = "wandb",
    run_name: str = "rc_pipeline",
    validation_split: Optional[float] = 0.05,
) -> None:
    if comprehension_type in [SynthMode.LLM, SynthMode.BOTH]:
        if not llm_kwargs:
            raise ValueError("llm_kwargs is required for LLM based generation")

    if comprehension_type in [SynthMode.REGEX, SynthMode.BOTH]:
        if not synth_kwargs:
            raise ValueError("synth_kwargs is required for regex based generation")

        if synth_kwargs and synth_kwargs.domain_spm_path:
            domain_spm = spm.SentencePieceProcessor(model_file=synth_kwargs.domain_spm_path)
        else:
            logger.warning("No domain tokenizer provided. The domain tokenizer will be created from the input files")
            domain_spm = create_domain_tokenizer_from_files(input, csv_column=csv_column)

        general_spm = spm.SentencePieceProcessor(model_file=synth_kwargs.general_spm_path)

    in_memory_dataset = []

    # generate regex based reading comprehension dataset
    if comprehension_type in [SynthMode.REGEX, SynthMode.BOTH]:
        # generate regex based reading comprehension dataset
        regex_rc_gen = RegexBasedReadingComprehension(general_spm, domain_spm)

        # NOTE: this is a simple check to see if the dataset is already generated
        in_memory_dataset.extend(
            [{"messages": rc_text} for _, _, rc_text in regex_rc_gen.dataset_generator(input, csv_column)]
        )

    # NOTE: this operation is time consuming and very expensive
    # Attention has been paid to try to save intermediate steps in case of failure
    # so that the generation can be resumed from the last checkpoint
    if comprehension_type in [SynthMode.LLM, SynthMode.BOTH] and llm_kwargs:
        if generation_state_file:
            if os.path.exists(generation_state_file):
                with open(generation_state_file, "rb") as f:
                    generation_state = pickle.load(f)
            else:
                generation_state = {"processed_files": [], "total_files": 0, "files_missed": 0}
                pickle.dump(generation_state, open(generation_state_file, "wb"))

        if not os.path.exists(llm_kwargs.dataset_output_path):
            os.makedirs(llm_kwargs.dataset_output_path)

        llm_rc_dataset_generator = generate_synthetic_dataset(
            model_name=llm_kwargs.model_name,
            input_directory_or_file=input,
            processed_files=generation_state["processed_files"],
            chunk=llm_kwargs.chunk or False,
            context_length=llm_kwargs.context_length or 0,
            csv_column=csv_column,
        )

        # generate llm based reading comprehension dataset
        for index, filename, context, gen_text in llm_rc_dataset_generator:
            qanda = question_and_answer_extractor(gen_text, context)
            if qanda:
                output_file = f"{filename}_{index}.json"
                with open(os.path.join(llm_kwargs.dataset_output_path, output_file), "w") as o:
                    json.dump(qanda, o)
            else:
                logger.warning(
                    (
                        f"No question and answer pairs found for {filename} " f"chunk: {index}"
                        if llm_kwargs.chunk
                        else ""
                    )
                )
                generation_state["files_missed"] += 1
            generation_state["processed_files"].append(filename)
            generation_state["total_files"] += 1
            pickle.dump(generation_state, open(generation_state_file, "wb"))

        logger.info(" Statistics ")
        success_files_count = generation_state["total_files"] - generation_state["files_missed"]
        logger.info(f"Total number of successfully extracted q&a: {success_files_count}")
        logger.info(f"Total files missed: {generation_state['files_missed']} out of {generation_state['total_files']}")
        logger.info(f"Total files processed: {generation_state['total_files']}")

        for file in os.listdir(llm_kwargs.dataset_output_path):
            with open(os.path.join(llm_kwargs.dataset_output_path, file), "r") as f:
                in_memory_dataset.append({"messages": json.load(f)})

    # shuffle in memory dataset
    random.shuffle(in_memory_dataset)

    dataset = datasets.Dataset.from_list(in_memory_dataset)

    if not os.path.exists(output_dataset_name):
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
        neftune_noise_alpha=neftune_noise_alpha,
        log_with=log_with,
        local_dataset=True,
        validation_split=validation_split,
        run_name=run_name,
    )

    if generation_state_file:
        os.remove(generation_state_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="HuggingFaceH4/zephyr-7b-alpha", help="name of the model to be trained"
    )
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
        "--llm_synth_model_name",
        type=str,
        default="HuggingFaceH4/zephyr-7b-beta",
        help="name of the model to be used for LLM based generation",
    )
    parser.add_argument(
        "--llm_synth_model_context_length", type=int, default=4096, help="context length to calulcate the chunk size"
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
        default="./resources/general.spm",
        help="path to the general tokenizer (needed for regex based generation)",
    )
    parser.add_argument(
        "--domain_spm_path",
        type=str,
        default=None,
        help="path to the domain tokenizer (needed for regex based generation)",
    )
    parser.add_argument("--input", type=str, required=True, help="A CSV file OR a directory containing the input files")
    parser.add_argument("--csv_column", type=str, help="Column to read from the CSV file")
    parser.add_argument("--no_chunk", action="store_true", help="whether to NOT chunk the input files or not")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="number of epochs to train the generator")
    parser.add_argument("--split", type=str, default="train", help="split to be used for training")
    parser.add_argument("--size_valid_set", type=int, default=1000, help="size of the validation set (STREAMING ONLY)")
    parser.add_argument("--validation_split", type=float, default=0.05, help="validation split")
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
    parser.add_argument("--lora_alpha", type=int, default=512, help="lora alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="lora dropout")
    parser.add_argument("--lora_r", type=int, default=256, help="lora r")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--optimizer_type", type=str, default="paged_adamw_32bit")
    parser.add_argument("--model_output_dir", type=str, default="model_output_dir")
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
    parser.add_argument("--run_name", type=str, default="rc_pipeline", help="name of the run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if os.path.isfile(args.input) and not args.csv_column:
        raise ValueError("a CSV column must be specified if the input is a file")

    llm_kwargs = None
    synth_kwargs = None

    if args.comprehension_type in [SynthMode.LLM, SynthMode.BOTH]:
        if not args.llm_synth_model_name:
            raise ValueError("llm_synth_model_name is required for LLM based generation")

        if not args.llm_dataset_output_path:
            raise ValueError("llm_dataset_output_path is required for LLM based generation")

        llm_kwargs = LLMKwargs(
            model_name=args.llm_synth_model_name,
            context_length=args.llm_synth_model_context_length,
            dataset_output_path=args.llm_dataset_output_path,
            chunk=not args.no_chunk,
        )

    if args.comprehension_type in [SynthMode.REGEX, SynthMode.BOTH]:
        if not args.general_spm_path:
            raise ValueError("general_spm_path is required for regex based generation")

        synth_kwargs = SynthKwargs(
            general_spm_path=args.general_spm_path,
            domain_spm_path=args.domain_spm_path,
        )

    pipeline(
        model_name=args.model_name,
        output_dataset_name=args.output_dataset_name,
        comprehension_type=args.comprehension_type,
        input=args.input,
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
        neftune_noise_alpha=args.neftune_noise_alpha,
        log_with=args.log_with,
        generation_state_file=args.generation_state_file,
        llm_kwargs=llm_kwargs,
        synth_kwargs=synth_kwargs,
        validation_split=args.validation_split,
        run_name=args.run_name,
        csv_column=args.csv_column,
    )


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()
