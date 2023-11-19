from typing import Optional
from enum import Enum
import os
import json
import datasets
import sentencepiece as spm

from dalm.datasets.reading_comprehension_generation.regex_based import RC
from dalm.datasets.reading_comprehension_generation.synthetic_based import generate_synthetic_dataset
from dalm.datasets.reading_comprehension_generation.utils import question_and_answer_extractor, text_chunker

from dalm.training.generator_only.trainer import train_generator


class SynthMode(Enum):
    REGEX = "regex"
    LLM = "llm"
    BOTH = "both"


def pipeline(
    model_name: str,
    comprehension_type: SynthMode = SynthMode.BOTH,
    llm_synth_model_name: Optional[str] = None,
    llm_synth_model_context_length: Optional[int] = 2048,  # TODO change
    general_spm_path: Optional[str] = None,
    domain_spm_path: Optional[str] = None,
    regex_dataset_output_path: Optional[str] = "regex_dataset",
    llm_dataset_output_path: Optional[str] = "llm_dataset",
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
):
    domain_spm = spm.SentencePieceProcessor(model_file=domain_spm_path)
    ori_spm = spm.SentencePieceProcessor(model_file=general_spm_path)

    # generate regex based reading comprehension dataset
    if comprehension_type in [SynthMode.REGEX, SynthMode.BOTH]:
        # generate regex based reading comprehension dataset
        regex_rc_gen = RC(ori_spm, domain_spm)

        # NOTE: this is a simple check to see if the dataset is already generated
        if not (os.path.exists(regex_dataset_output_path) and len(os.listdir(regex_dataset_output_path)) > 0):
            regex_rc_gen.create_dataset(
                dataset_path,
                regex_dataset_output_path,
            )

    # generate llm based reading comprehension dataset
    if comprehension_type in [SynthMode.LLM, SynthMode.BOTH]:
        # generate llm based reading comprehension dataset
        for index, (gen_text, context) in enumerate(
            generate_synthetic_dataset(
                model_name=llm_synth_model_name,
                input_directory=dataset_path,
                state_file=generation_state_file,
                chunk=chunk,
                context_length=llm_synth_model_context_length,
            )
        ):
            qanda = question_and_answer_extractor(gen_text, context)
            if qanda:
                output_file = f"gen_{index}.json"
                with open(os.path.join(llm_dataset_output_path, output_file), "w") as o:
                    json.dump(qanda, o)

    # mix both and make it a huggingface dataset
    list_of_data = []
    if comprehension_type in [SynthMode.REGEX, SynthMode.BOTH]:
        # a1 = load_dataset("json", data_dir=regex_dataset_output_path) This does not work
        for file in os.listdir(regex_dataset_output_path):
            text = json.load(open(os.path.join(regex_dataset_output_path, file), "r"))
            list_of_data.append({"messages": text})

    if comprehension_type in [SynthMode.LLM, SynthMode.BOTH]:
        # a2 = load_dataset("json", data_dir=llm_dataset_output_path) This does not work
        for file in os.listdir(llm_dataset_output_path):
            text = json.load(open(os.path.join(llm_dataset_output_path, file), "r"))
            list_of_data.append({"messages": text})

    if comprehension_type == SynthMode.BOTH:
        dataset = datasets.Dataset.from_list(list_of_data)

    dataset.save_to_disk("reading_comprehension_dataset")  # TODO: change name from

    # del dataset # TODO: change name

    train_generator(
        model_name=model_name,
        dataset_name="reading_comprehension_dataset",
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
    )


if __name__ == "__main__":
    pipeline(
        "HuggingFaceH4/zephyr-7b-beta",
        comprehension_type=SynthMode.BOTH,
        llm_synth_model_name="HuggingFaceH4/zephyr-7b-beta",
        domain_spm_path="./tokenizers/domain.spm",
        general_spm_path="./tokenizers/general.spm",
        chunk=True,
        dataset_path="./data_llm",
        packing=True,
    )
