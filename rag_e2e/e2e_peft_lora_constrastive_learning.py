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

import datasets
import evaluate
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import transformers.utils.logging

from transformers import GPT2Tokenizer
from peft import LoraConfig, TaskType, get_peft_model
from base_model import AutoModelForSentenceEmbedding
from train_utils import (
    save_model_hook,
    load_model_hook,
    get_cosine_sim,
    get_nt_xent_loss,
)
from trl import AutoModelForCausalLMWithRAG, RagE2EConfig, ArceeRagTrainer

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training a PEFT model for Sematic Search task"
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="dataset path in the local dir"
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
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--logit_scale",
        type=int,
        default=100,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
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
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_token", type=str, help="The token to use to push to the Model Hub."
    )
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


def main():
    args = parse_args()
    # get the retriever tokenizer
    r_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # Get the causal tokenizer
    c_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    c_tokenizer.pad_token = c_tokenizer.eos_token

    # base retriever model
    r_model = AutoModelForSentenceEmbedding(args.model_name_or_path, r_tokenizer)
    # base causal model
    c_model = AutoModelForCausalLMWithRAG.from_pretrained("gpt2")

    if args.use_peft:
        # peft config and wrapping
        peft_dict = dict(
            r=8,
            lora_alpha=16,
            bias="none",
            target_modules=["key", "query", "value"],
        )
        retriever_config = peft_dict.copy()
        retriever_config["task_type"] = TaskType.FEATURE_EXTRACTION
        r_peft_config = LoraConfig(**retriever_config)
        r_model = get_peft_model(r_model, r_peft_config)
        r_model.print_trainable_parameters()

        c_config = peft_dict.copy()
        c_config["task_type"] = TaskType.CAUSAL_LM
        c_peft_config = LoraConfig(**c_config)
        c_model = get_peft_model(c_model, c_peft_config)
        c_model.print_trainable_parameters()

    rag_config = {"batch_size": 1}
    config = RagE2EConfig(**rag_config)
    arcee_rag_trainer = ArceeRagTrainer(config, c_model, c_tokenizer)
    accelerator = arcee_rag_trainer.accelerator
    accelerator.print(r_model)

    # TODO: Can I replace the accelerator with the one here?
    accelerator = arcee_rag_trainer.accelerator

    # accelerator = (
    #     Accelerator(log_with=args.report_to, project_dir=args.output_dir)
    #     if args.with_tracking
    #     else Accelerator()
    # )
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


    # dataset download and preprocessing

    # dataset = datasets.load_dataset("csv", data_files={"train": f"{args.dataset_path}/train.csv",
    #                                                    "validation": f"{args.dataset_path}/valid.csv"})
    dataset = datasets.load_dataset(
        "csv",
        data_files={
            "train": "triplets.csv",
            "validation": "triplets.csv",
        },
    )

    def preprocess_function(examples):
        queries = examples["query"]
        passages = examples["passage"]
        answers = examples["answer"]
        # Tokenize for Retriever
        r_input = r_tokenizer(
            queries, padding="max_length", max_length=512, truncation=True
        )
        r_output = r_tokenizer(
            passages, padding="max_length", max_length=512, truncation=True
        )

        # Tokenize for causal model
        # Here, we need to combine the query and passage as the input, and the answer as the output
        inputs = [
            f"###passage### {passage}\n\n###query### {query}"
            for passage, query in zip(passages, queries)
        ]
        c_input = c_tokenizer(
            inputs, padding="max_length", max_length=1024, truncation=True
        )
        c_output = c_tokenizer(
            answers, padding="max_length", max_length=1024, truncation=True
        )

        pre_batch = {}
        for k, v in r_input.items():
            pre_batch[f"r_input_{k}"] = v
        for k, v in r_output.items():
            pre_batch[f"r_output_{k}"] = v
        for k, v in c_input.items():
            pre_batch[f"c_input_{k}"] = v
        for k, v in c_output.items():
            pre_batch[f"c_output_{k}"] = v
        return pre_batch

    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Running tokenizer on dataset",
        num_proc=4,
    )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(processed_datasets["train"])), 3):
        logger.info(
            f"Sample {index} of the training set: {processed_datasets['train'][index]}."
        )

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

    r_optimizer = torch.optim.Adam(r_model.parameters(), lr=args.learning_rate)
    c_optimizer = torch.optim.Adam(c_model.parameters(), lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    r_lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=r_optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    c_lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=c_optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    # see https://github.com/huggingface/accelerate/issues/253#issuecomment-1253231210
    r_model, c_model = accelerator.prepare(r_model, c_model)
    r_optimizer, c_optimizer, r_lr_scheduler, c_lr_scheduler = accelerator.prepare(
        r_optimizer, c_optimizer, r_lr_scheduler, c_lr_scheduler
    )

    # (
    #     model,
    #     optimizer,
    #     train_dataloader,
    #     eval_dataloader,
    #     lr_scheduler,
    # ) = accelerator.prepare(
    #     r_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    # )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
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
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        accelerator.init_trackers("peft_contrastive_learning", experiment_config)

    metric = evaluate.load("accuracy")

    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    if args.use_peft:
        # saving and loading checkpoints for resuming training
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(processed_datasets['train'])}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
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
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_step

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        r_model.train()
        if args.with_tracking:
            total_loss = 0
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(r_model, c_model):
                query_embs = r_model(
                    **{
                        k.replace("r_input_", ""): v
                        for k, v in batch.items()
                        if "r_input" in k
                    }
                )
                passage_embs = r_model(
                    **{
                        k.replace("r_output_", ""): v
                        for k, v in batch.items()
                        if "r_output" in k
                    }
                )
                logits = get_cosine_sim(query_embs, passage_embs, args.logit_scale)
                loss_query = get_nt_xent_loss(logits)
                loss_passage = get_nt_xent_loss(logits.t())
                # Retriever loss
                r_loss = (loss_query + loss_passage) / 2.0

                # Get the loss for the causal model
                # 8. combine the prompt with the answer for  casual llm training
                query_answer_inputs = torch.stack([
                    query_input + answer_input for query_input, answer_input in
                    zip(batch["c_input_ids"], batch["c_output_input_ids"])
                ], dim=0)
                query_answer_am = torch.stack([
                    query_input + answer_input for query_input, answer_input in
                    zip(batch["c_input_attention_mask"], batch["c_output_attention_mask"])
                ], dim=0)
                prompt_tokens = torch.tensor(batch["c_input_ids"])
                c_logits = c_model(input_ids=query_answer_inputs, attention_mask=query_answer_am)

                # query_passage_logits = r_model(
                #     **{
                #         k.replace("c_input_", ""): v
                #         for k, v in batch.items()
                #         if "c_input" in k
                #     }
                # )
                # answer_logits = r_model(
                #     **{
                #         k.replace("c_output_", ""): v
                #         for k, v in batch.items()
                #         if "c_output" in k
                #     }
                # )
                # 6. train model with ppo
                marginalize_casual_loss = arcee_rag_trainer.compute_marginalized_loss_from_logits(
                    c_logits,
                    query_answer_inputs,
                    r_loss.unsqueeze(0),
                    prompt_tokens
                )

                total_loss += accelerator.reduce(marginalize_casual_loss.detach().float(), reduction="sum")
                accelerator.backward(marginalize_casual_loss)
                r_optimizer.step()
                r_lr_scheduler.step()
                r_model.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if (step + 1) % 100 == 0:
                logger.info(f"Step: {step+1}, Loss: {total_loss/(step+1)}")
                if args.with_tracking:
                    accelerator.log(
                        {"train/loss": total_loss / (step + 1)}, step=completed_steps
                    )

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        # for the efficiency we do a per-batch evaluation
        r_model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                query_embs = r_model(
                    **{
                        k.replace("query_", ""): v
                        for k, v in batch.items()
                        if "query" in k
                    }
                )
                passage_embs = r_model(
                    **{
                        k.replace("passage_", ""): v
                        for k, v in batch.items()
                        if "passage" in k
                    }
                )
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
            result["train/epoch_loss"] = total_loss.item() / len(train_dataloader)
            accelerator.log(result, step=completed_steps)

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if isinstance(checkpointing_steps, str):
                    accelerator.save_state(
                        os.path.join(args.output_dir, f"epoch_{epoch}")
                    )
                accelerator.unwrap_model(r_model).save_pretrained(
                    args.output_dir,
                    state_dict=accelerator.get_state_dict(
                        accelerator.unwrap_model(r_model)
                    ),
                )
                r_tokenizer.save_pretrained(args.output_dir)
            accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()


# python rag_e2e/e2e_peft_lora_constrastive_learning.py  --dataset_path "./dataset" --model_name_or_path "BAAI/bge-small-en" --output_dir "./contrastive_checkpoints" --use_peft  --with_tracking --report_to tensorboard
