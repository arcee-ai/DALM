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

import datasets
import evaluate
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from black import Union
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, SchedulerType, default_data_collator, get_scheduler

from dalm.models.retriever_only_base_model import AutoModelForSentenceEmbedding
from dalm.training.utils.retriever_only_training_utils import preprocess_dataset
from dalm.training.utils.train_utils import get_cosine_sim, get_nt_xent_loss, load_model_hook, save_model_hook

logger = get_logger(__name__)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="training a PEFT model for Sematic Search task")
    parser.add_argument("--train_dataset_csv_path", type=str, default=None, help="dataset path in the local dir")
    parser.add_argument("--test_dataset_csv_path", type=str, default=None, help="dataset path in the local dir")
    parser.add_argument(
        "--dataset_query_col_name", type=str, default="Question", help="Name of the query column in the dataset"
    )
    parser.add_argument(
        "--dataset_passage_col_name", type=str, default="Abstract", help="Name of the passage column in the dataset"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
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
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
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
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
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
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
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


def main() -> None:
    args = parse_args()
    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
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
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # dataset download and preprocessing

    dataset = datasets.load_dataset(
        "csv", data_files={"train": args.train_dataset_csv_path, "validation": args.test_dataset_csv_path}
    )

    processed_datasets = dataset.map(
        lambda example: preprocess_dataset(
            example,
            tokenizer,
            query_col_name=args.dataset_query_col_name,
            passage_col_name=args.dataset_passage_col_name,
            max_length=args.max_length,
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(processed_datasets["train"])), 3):
        logger.info(f"Sample {index} of the training set: {processed_datasets['train'][index]}.")

    # base model
    model = AutoModelForSentenceEmbedding(args.model_name_or_path, tokenizer, use_bnb=True, get_peft=args.use_peft)
    model.print_trainable_parameters()  # type: ignore # No idea what mypy is complaining about.

    accelerator.print(model)

    # get dataloaders
    train_dataloader = DataLoader(
        processed_datasets["train"],
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
        pin_memory=True,
    )

    eval_dataloader = DataLoader(
        processed_datasets["validation"],
        shuffle=False,
        collate_fn=default_data_collator,
        batch_size=args.per_device_eval_batch_size,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("peft_contrastive_learning", experiment_config)

    metric = evaluate.load("accuracy")

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if args.use_peft:
        # saving and loading checkpoints for resuming training
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(processed_datasets['train'])}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
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
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_step

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss: Union[float, torch.Tensor] = 0.0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                query_embs = model(**{k.replace("query_", ""): v for k, v in batch.items() if "query" in k})
                passage_embs = model(**{k.replace("passage_", ""): v for k, v in batch.items() if "passage" in k})
                logits = get_cosine_sim(query_embs, passage_embs, args.logit_scale)

                loss_query = get_nt_xent_loss(logits)
                loss_passage = get_nt_xent_loss(logits.t())

                loss = (loss_query + loss_passage) / 2.0
                total_loss += accelerator.reduce(loss.detach().float(), reduction="sum")
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if (step + 1) % 100 == 0:
                logger.info(f"Step: {step+1}, Loss: {total_loss/(step+1)}")
                if args.with_tracking:
                    accelerator.log({"train/loss": total_loss / (step + 1)}, step=completed_steps)

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        # for the efficiency we do a per-batch evaluation
        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                query_embs = model(**{k.replace("query_", ""): v for k, v in batch.items() if "query" in k})
                passage_embs = model(**{k.replace("passage_", ""): v for k, v in batch.items() if "passage" in k})
                logits = get_cosine_sim(query_embs, passage_embs, args.logit_scale)
                # we just need an identity matrix
                labels = torch.arange(len(logits), device=logits.device)
            logits, labels = accelerator.gather_for_metrics((logits, labels))
            metric.add_batch(
                predictions=torch.argmax(logits, dim=1),
                references=labels,
            )

        result = metric.compute()
        result = {f"eval/{k}": v for k, v in result.items()}
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:", result)
        if args.with_tracking:
            step_loss = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
            result["train/epoch_loss"] = step_loss / len(train_dataloader)
            accelerator.log(result, step=completed_steps)

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if isinstance(checkpointing_steps, str):
                    accelerator.save_state(os.path.join(args.output_dir, f"epoch_{epoch}"))
                accelerator.unwrap_model(model).save_pretrained(
                    args.output_dir, state_dict=accelerator.get_state_dict(accelerator.unwrap_model(model))
                )
                tokenizer.save_pretrained(args.output_dir)
            accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()


# python contrastive_train/peft_lora_constrastive_learning.py  --train_dataset_csv_path "xxxx.csv" \
#     --test_dataset_csv_path "yyyy.csv" \
#     --model_name_or_path "BAAI/bge-small-en" --output_dir "./contrastive_checkpoints" --use_peft  \
#     --with_tracking --report_to tensorboard
