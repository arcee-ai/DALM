# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import random
from argparse import Namespace
from types import NoneType
from typing import Dict, Optional, Union

import datasets
import torch
import transformers
import transformers.utils.logging
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from dalm.models.rag_e2e_base_model import AutoModelForRagE2E
from dalm.training.utils.rag_e2e_dataloader_utils import preprocess_dataset
from dalm.training.utils.train_utils import (
    compute_marginalized_loss_from_logits,
    get_cosine_sim,
    get_nt_xent_loss,
    load_model_hook,
    save_model_hook,
)
from dalm.utils import load_dataset

logger = get_logger(__name__)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="training a PEFT model for Sematic Search task")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help=("Dataset path. Can be a huggingface dataset directory or a csv file."),
    )
    parser.add_argument(
        "--passage_column_name", type=str, default="Abstract", help="Name of the column containing the passage"
    )
    parser.add_argument(
        "--query_column_name", type=str, default="Question", help="Name of the column containing the query"
    )
    parser.add_argument(
        "--answer_column_name", type=str, default="Answer", help="Name of the column containing the answer"
    )
    parser.add_argument(
        "--query_max_len",
        type=int,
        default=50,
        help=(
            "The maximum total query sequence length after tokenization. Sequences longer than this will be truncated,"
        ),
    )
    parser.add_argument(
        "--passage_max_len",
        type=int,
        default=128,
        help=(
            "The maximum total passage sequence length after tokenization. "
            "Sequences longer than this will be truncated,"
        ),
    )
    parser.add_argument(
        "--generator_max_len",
        type=int,
        default=256,
        help=(
            "The maximum total generator input sequence length after tokenization. "
            "Sequences longer than this will be truncated,"
        ),
    )
    parser.add_argument(
        "--retriever_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--generator_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--logit_scale",
        type=int,
        default=100,
        help="logit scale for the constrastive learning.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            SchedulerType.LINEAR,
            SchedulerType.COSINE,
            SchedulerType.COSINE_WITH_RESTARTS,
            SchedulerType.POLYNOMIAL,
            SchedulerType.CONSTANT,
            SchedulerType.CONSTANT_WITH_WARMUP,
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`, `"wandb"`, '
            '`"mlflow"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--sanity_test",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--use_peft",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    args = parser.parse_args()

    return args


def train_e2e(
    dataset_or_path: str | Dataset,
    retriever_name_or_path: str,
    generator_name_or_path: str,
    passage_column_name: str = "Abstract",
    query_column_name: str = "Question",
    answer_column_name: str = "Answer",
    query_max_len: int = 50,
    passage_max_len: int = 128,
    generator_max_len: int = 256,
    per_device_train_batch_size: int = 32,
    learning_rate: float = 1e-4,
    logit_scale: int = 100,
    weight_decay: float = 0.0,
    num_train_epochs: int = 1,
    max_train_steps: Optional[int] = None,
    gradient_accumulation_steps: int = 1,
    lr_scheduler_type: SchedulerType = SchedulerType.LINEAR,
    num_warmup_steps: int = 100,
    output_dir: Optional[str] = None,
    seed: int = 42,
    hub_model_id: Optional[str] = None,
    hub_token: Optional[str] = None,
    checkpointing_steps: Optional[int | str] = None,
    resume_from_checkpoint: Optional[str] = None,
    with_tracking: bool = True,
    report_to: str = "all",
    sanity_test: bool = True,
    use_peft: bool = True,
) -> None:
    """Train an in-domain model e2e with a retriever and generator. See `dalm train-rag-e2e --help` for more details"""
    # Get the passed in vars before beginning training, in case we report training
    args = dict(locals())
    # TensorBoard cannot log Enums, need the raw value
    args["lr_scheduler_type"] = args["lr_scheduler_type"].value
    args = {k: v for k, v in args.items() if v is None or isinstance(v, (float, int, str, NoneType))}
    # rag retriver and the generator
    rag_model = AutoModelForRagE2E(retriever_name_or_path, generator_name_or_path)

    accelerator = Accelerator(log_with=report_to, project_dir=output_dir) if with_tracking else Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # dataset download and preprocessing
    dataset = load_dataset(dataset_or_path)
    retriever_tokenizer = rag_model.retriever_tokenizer
    generator_tokenizer = rag_model.generator_tokenizer
    generator_tokenizer.pad_token = generator_tokenizer.eos_token

    processed_datasets = dataset.map(
        lambda example: preprocess_dataset(
            example,
            retriever_tokenizer=rag_model.retriever_tokenizer,
            generator_tokenizer=rag_model.generator_tokenizer,
            query_column_name=query_column_name,
            passage_column_name=passage_column_name,
            answer_column_name=answer_column_name,
            query_max_len=query_max_len,
            passage_max_len=passage_max_len,
            generator_max_len=generator_max_len,
        ),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on dataset",
        num_proc=1,
    )
    # Log a few random samples from the training set:
    for index in random.sample(range(len(processed_datasets)), 2):
        logger.info(f"Sample {index} of the training set: {processed_datasets[index]}.")

    # get dataloaders
    train_dataloader = DataLoader(
        processed_datasets,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=per_device_train_batch_size,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(rag_model.parameters(), lr=learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    (rag_model, optimizer, train_dataloader, lr_scheduler) = accelerator.prepare(
        rag_model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = checkpointing_steps
    if checkpointing_steps is not None and str(checkpointing_steps).isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if with_tracking:
        experiment_config = args.copy()
        accelerator.init_trackers("peft_rag_e2e_learning", experiment_config)

    total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    if use_peft:
        # saving and loading checkpoints for resuming training
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(processed_datasets)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint is not None or resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {resume_from_checkpoint}")
            accelerator.load_state(resume_from_checkpoint)
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // gradient_accumulation_steps

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, num_train_epochs):
        rag_model.train()
        total_loss: Union[float, torch.Tensor] = 0.0
        if resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(rag_model):
                query_embs = rag_model(
                    "retrieval", batch["retriever_query_input_ids"], batch["retriever_query_attention_mask"]
                )

                passage_embs = rag_model(
                    "retrieval",
                    batch["retriever_passage_input_ids"],
                    batch["retriever_passage_attention_mask"],
                )

                logits = get_cosine_sim(query_embs, passage_embs, logit_scale)

                loss_query = get_nt_xent_loss(logits)
                loss_passage = get_nt_xent_loss(logits.t())
                # Retriever loss
                retriver_contrastive_loss = (loss_query + loss_passage) / 2.0

                # Get the loss for the causal model
                # 8. combine the prompt with the answer for  casual llm training

                ### add the loss casual here

                generator_logits = rag_model(
                    "generation", batch["generator_input_input_ids"], batch["generator_input_attention_mask"]
                )

                marginalize_casual_loss = compute_marginalized_loss_from_logits(
                    generator_logits,
                    batch["generator_input_input_ids"],
                    batch["generator_input_attention_mask"],
                    # assume since we know what is matching
                    # TODO: test this also by taking the torch.argmax(logits, dim=1)
                    logits,
                    batch["query_passage_input_len"],
                )

                combined_loss = retriver_contrastive_loss + marginalize_casual_loss

                total_loss += accelerator.reduce(combined_loss.detach().float(), reduction="sum")

                accelerator.backward(combined_loss)
                optimizer.step()
                lr_scheduler.step()
                rag_model.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if (step + 1) % 100 == 0:
                logger.info(f"Step: {step + 1}, Loss: {total_loss / (step + 1)}")
                if with_tracking:
                    accelerator.log({"train/loss": total_loss / (step + 1)}, step=completed_steps)

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    step_output_dir = f"step_{completed_steps}"
                    if output_dir is not None:
                        output_dir = os.path.join(output_dir, step_output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= max_train_steps:
                break

        result: Dict[str, Union[int, float, torch.Tensor]] = {}
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:", result)
        if with_tracking:
            step_loss = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
            result["train/epoch_loss"] = step_loss / len(train_dataloader)
            accelerator.log(result, step=completed_steps)

        if output_dir is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if isinstance(checkpointing_steps, str):
                    accelerator.save_state(os.path.join(output_dir, f"epoch_{epoch}"))

                retriever_ckpt_path = output_dir + "/retriever"
                generator_ckpt_path = output_dir + "/generator"

                # retriever saving
                unwrapped_rag_model = accelerator.unwrap_model(rag_model)
                unwrapped_rag_model.retriever_model.save_pretrained(
                    retriever_ckpt_path,
                    state_dict=accelerator.get_state_dict(unwrapped_rag_model.retriever_model),
                )
                retriever_tokenizer.save_pretrained(retriever_ckpt_path)

                # generator saving
                unwrapped_rag_model.generator_model.save_pretrained(
                    generator_ckpt_path,
                    state_dict=accelerator.get_state_dict(unwrapped_rag_model.generator_model),
                )
                generator_tokenizer.save_pretrained(generator_ckpt_path)
            accelerator.wait_for_everyone()
    if with_tracking:
        accelerator.end_training()


def main() -> None:
    args = parse_args()
    train_e2e(
        dataset_or_path=args.dataset_path,
        retriever_name_or_path=args.retriever_name_or_path,
        generator_name_or_path=args.generator_name_or_path,
        passage_column_name=args.passage_column_name,
        query_column_name=args.query_column_name,
        answer_column_name=args.answer_column_name,
        query_max_len=args.query_max_len,
        passage_max_len=args.passage_max_len,
        generator_max_len=args.generator_max_len,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logit_scale=args.logit_scale,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_train_steps=args.max_train_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        num_warmup_steps=args.num_warmup_steps,
        output_dir=args.output_dir,
        seed=args.seed,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
        checkpointing_steps=args.checkpointing_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        with_tracking=args.with_tracking,
        report_to=args.report_to,
        sanity_test=args.sanity_test,
        use_peft=args.use_peft,
    )


if __name__ == "__main__":
    main()

# python dalm/training/rag_e2e/train_rage2e.py
#   --dataset_path "/root/DALM/dataset/out/question_answer_pairs_train"
#   --retriever_name_or_path "BAAI/bge-large-en"
#   --generator_name_or_path "meta-llama/Llama-2-7b-hf"
#   --output_dir "./rag_e2e_checkpoints"
#   --with_tracking
#   --report_to all
#   --per_device_train_batch_size 32
