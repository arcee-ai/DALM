from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from transformers import SchedulerType
from typing_extensions import Annotated

from dalm import __version__
from dalm.datasets.qa_gen.question_answer_generation import generate_qa_from_disk
from dalm.eval.eval_rag import evaluate_rag
from dalm.eval.eval_retriever_only import evaluate_retriever
from dalm.training.rag_e2e.train_rage2e import train_e2e
from dalm.training.retriever_only.train_retriever_only import train_retriever

cli = typer.Typer()
HERE = Path(__file__).parent


class DALMSchedulerType(Enum):
    LINEAR = SchedulerType.LINEAR
    COSINE = SchedulerType.COSINE
    COSINE_WITH_RESTARTS = SchedulerType.COSINE_WITH_RESTARTS
    POLYNOMIAL = SchedulerType.POLYNOMIAL
    CONSTANT = SchedulerType.CONSTANT
    CONSTANT_WITH_WARMUP = SchedulerType.CONSTANT_WITH_WARMUP


class TorchDtype(str, Enum):
    float16 = "float16"
    bfloat16 = "bfloat16"


@cli.command()
def version() -> None:
    """Print the current version of DALM"""
    print(f"🐾You are running DALM version: {__version__}")


@cli.command()
def train_rag_e2e(
    dataset_path: Annotated[
        str,
        typer.Argument(
            help="Path to the dataset to train with. Can be an hf dataset dir, csv file, or path to hub file.",
            show_default=False,
        ),
    ],
    retriever_name_or_path: Annotated[
        str,
        typer.Argument(
            help="Path to pretrained retriever or identifier from huggingface.co/models.", show_default=False
        ),
    ],
    generator_name_or_path: Annotated[
        str,
        typer.Argument(
            help="Path to pretrained (causal) generator or identifier from huggingface.co/models.", show_default=False
        ),
    ],
    passage_column_name: Annotated[str, typer.Option(help="Name of the column containing the passage")] = "Abstract",
    query_column_name: Annotated[str, typer.Option(help="Name of the column containing the query")] = "Question",
    answer_column_name: Annotated[str, typer.Option(help="Name of the column containing the Answer")] = "Answer",
    query_max_len: Annotated[
        int, typer.Option(help="The max query sequence length during tokenization. Longer sequences are truncated")
    ] = 50,
    passage_max_len: Annotated[
        int, typer.Option(help="The max passage sequence length during tokenization. Longer sequences are truncated")
    ] = 128,
    generator_max_len: Annotated[
        int,
        typer.Option(
            help="The max generator input sequence length during tokenization. Longer sequences are truncated"
        ),
    ] = 256,
    per_device_train_batch_size: Annotated[
        int, typer.Option(help="Batch size (per device) for the training dataloader.")
    ] = 32,
    learning_rate: Annotated[
        float, typer.Option(help="Initial learning rate (after the potential warmup period) to use.")
    ] = 1e-4,
    logit_scale: Annotated[int, typer.Option(help="Logit scale for the contrastive learning.")] = 100,
    weight_decay: Annotated[float, typer.Option(help="Weight decay to use.")] = 0.0,
    num_train_epochs: Annotated[int, typer.Option(help="Total number of training epochs to perform.")] = 1,
    max_train_steps: Annotated[
        Optional[int],
        typer.Option(help="Total number of training steps to perform. If provided, overrides num_train_epochs."),
    ] = None,
    gradient_accumulation_steps: Annotated[
        int, typer.Option(help="Number of updates steps to accumulate before performing a backward/update pass.")
    ] = 1,
    lr_scheduler_type: Annotated[
        DALMSchedulerType, typer.Option(help="The scheduler type to use.")
    ] = DALMSchedulerType.LINEAR.value,
    num_warmup_steps: Annotated[int, typer.Option(help="Number of steps for the warmup in the lr scheduler.")] = 100,
    output_dir: Annotated[Optional[str], typer.Option(help="Where to store the final model.")] = None,
    seed: Annotated[int, typer.Option(help="A seed for reproducible training.")] = 42,
    hub_model_id: Annotated[
        Optional[str],
        typer.Option(
            help="[NOT CURRENTLY USED]. The name of the repository to keep in sync with the local `output_dir`."
        ),
    ] = None,
    hub_token: Annotated[
        Optional[str], typer.Option(help="[NOT CURRENTLY USED]. The token to use to push to the Model Hub.")
    ] = None,
    checkpointing_steps: Annotated[
        Optional[str],
        typer.Option(
            help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."
        ),
    ] = None,
    resume_from_checkpoint: Annotated[
        Optional[str], typer.Option(help="If the training should continue from a checkpoint folder.")
    ] = None,
    with_tracking: Annotated[bool, typer.Option(help="Whether to enable experiment trackers for logging.")] = True,
    report_to: Annotated[
        str,
        typer.Option(
            help=(
                'The integration to report the results and logs to. Supported platforms are `"tensorboard"`, '
                '`"wandb"`, `"mlflow"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all.'
                "integrations. Only applicable when `--with_tracking` is passed."
            )
        ),
    ] = "all",
    sanity_test: Annotated[
        bool, typer.Option(help="[NOT CURRENTLY USED]. Whether to sanity test the model after training")
    ] = True,
    use_peft: Annotated[bool, typer.Option(help="Whether to use Peft during fine-tuning.")] = True,
) -> None:
    """End-to-end train an in-domain model, including the retriever and generator"""
    train_e2e(
        dataset_or_path=dataset_path,
        retriever_name_or_path=retriever_name_or_path,
        generator_name_or_path=generator_name_or_path,
        passage_column_name=passage_column_name,
        query_column_name=query_column_name,
        answer_column_name=answer_column_name,
        query_max_len=query_max_len,
        passage_max_len=passage_max_len,
        generator_max_len=generator_max_len,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        logit_scale=logit_scale,
        weight_decay=weight_decay,
        num_train_epochs=num_train_epochs,
        max_train_steps=max_train_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type=lr_scheduler_type.value,
        num_warmup_steps=num_warmup_steps,
        output_dir=output_dir,
        seed=seed,
        hub_model_id=hub_model_id,
        hub_token=hub_token,
        checkpointing_steps=checkpointing_steps,
        resume_from_checkpoint=resume_from_checkpoint,
        with_tracking=with_tracking,
        report_to=report_to,
        sanity_test=sanity_test,
        use_peft=use_peft,
    )


@cli.command()
def train_retriever_only(
    retriever_name_or_path: Annotated[
        str, typer.Argument(help="Path to the model or identifier from huggingface.co/models.", show_default=False)
    ],
    dataset_path: Annotated[
        str,
        typer.Argument(
            help="Path to the train dataset to train with. Can be an hf dataset dir, csv file, or path to hub file.",
            show_default=False,
        ),
    ],
    passage_column_name: Annotated[str, typer.Option(help="Name of the column containing the passage")] = "Abstract",
    query_column_name: Annotated[str, typer.Option(help="Name of the column containing the query")] = "Question",
    query_max_len: Annotated[
        int, typer.Option(help="The max query sequence length during tokenization. Longer sequences are truncated")
    ] = 50,
    passage_max_len: Annotated[
        int, typer.Option(help="The max passage sequence length during tokenization. Longer sequences are truncated")
    ] = 128,
    per_device_train_batch_size: Annotated[
        int, typer.Option(help="Batch size (per device) for the training dataloader.")
    ] = 32,
    learning_rate: Annotated[
        float, typer.Option(help="Initial learning rate (after the potential warmup period) to use.")
    ] = 1e-4,
    logit_scale: Annotated[int, typer.Option(help="Logit scale for the contrastive learning.")] = 100,
    weight_decay: Annotated[float, typer.Option(help="Weight decay to use.")] = 0.0,
    num_train_epochs: Annotated[int, typer.Option(help="Total number of training epochs to perform.")] = 3,
    max_train_steps: Annotated[
        Optional[int],
        typer.Option(help="Total number of training steps to perform. If provided, overrides num_train_epochs."),
    ] = None,
    gradient_accumulation_steps: Annotated[
        int, typer.Option(help="Number of updates steps to accumulate before performing a backward/update pass.")
    ] = 1,
    lr_scheduler_type: Annotated[
        DALMSchedulerType, typer.Option(help="The scheduler type to use.")
    ] = DALMSchedulerType.LINEAR.value,
    num_warmup_steps: Annotated[int, typer.Option(help="Number of steps for the warmup in the lr scheduler.")] = 0,
    output_dir: Annotated[Optional[str], typer.Option(help="Where to store the final model.")] = None,
    seed: Annotated[int, typer.Option(help="A seed for reproducible training.")] = 42,
    hub_model_id: Annotated[
        Optional[str],
        typer.Option(
            help="[NOT CURRENTLY USED]. The name of the repository to keep in sync with the local `output_dir`."
        ),
    ] = None,
    hub_token: Annotated[
        Optional[str], typer.Option(help="[NOT CURRENTLY USED]. The token to use to push to the Model Hub.")
    ] = None,
    checkpointing_steps: Annotated[
        Optional[str],
        typer.Option(
            help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."
        ),
    ] = None,
    resume_from_checkpoint: Annotated[
        Optional[str], typer.Option(help="If the training should continue from a checkpoint folder.")
    ] = None,
    with_tracking: Annotated[bool, typer.Option(help="Whether to enable experiment trackers for logging.")] = True,
    report_to: Annotated[
        str,
        typer.Option(
            help=(
                'The integration to report the results and logs to. Supported platforms are `"tensorboard"`, '
                '`"wandb"`, `"mlflow"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all.'
                "integrations. Only applicable when `--with_tracking` is passed."
            )
        ),
    ] = "all",
    sanity_test: Annotated[
        bool, typer.Option(help="[NOT CURRENTLY USED]. Whether to sanity test the model after training")
    ] = True,
    use_peft: Annotated[bool, typer.Option(help="Whether to use Peft during fine-tuning.")] = True,
) -> None:
    """Train only the retriever using contrastive training"""
    train_retriever(
        dataset_or_path=dataset_path,
        retriever_name_or_path=retriever_name_or_path,
        passage_column_name=passage_column_name,
        query_column_name=query_column_name,
        query_max_len=query_max_len,
        passage_max_len=passage_max_len,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        logit_scale=logit_scale,
        weight_decay=weight_decay,
        num_train_epochs=num_train_epochs,
        max_train_steps=max_train_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type=lr_scheduler_type.value,
        num_warmup_steps=num_warmup_steps,
        output_dir=output_dir,
        seed=seed,
        hub_model_id=hub_model_id,
        hub_token=hub_token,
        checkpointing_steps=checkpointing_steps,
        resume_from_checkpoint=resume_from_checkpoint,
        with_tracking=with_tracking,
        report_to=report_to,
        sanity_test=sanity_test,
        use_peft=use_peft,
    )


@cli.command()
def qa_gen(
    dataset_path: Annotated[
        str,
        typer.Argument(
            help="Path to the input dataset. Can be huggingface dataset directory, "
            "path to a dataset on hub, or a csv file.",
            show_default=False,
        ),
    ],
    output_dir: Annotated[str, typer.Option(help="Output directory to store the resulting files")] = str(HERE),
    passage_column_name: Annotated[str, typer.Option(help="Column name for the passage/text")] = "Abstract",
    title_column_name: Annotated[str, typer.Option(help="Column name for the title of the full document")] = "Title",
    batch_size: Annotated[
        int, typer.Option(help="Batch size (per device) for generating question answer pairs.")
    ] = 100,
    sample_size: Annotated[
        int, typer.Option(help="Number of examples to process. If the data has more samples, they will be dropped")
    ] = 1000,
    as_csv: Annotated[
        bool,
        typer.Option(
            help="Save the files as CSV. If False, will save them as a dataset directory via [`~Dataset.save_to_disk`]"
        ),
    ] = True,
) -> None:
    """Generate question-answer pairs for a given input dataset"""
    generate_qa_from_disk(
        dataset_path, passage_column_name, title_column_name, sample_size, batch_size, output_dir, as_csv
    )


@cli.command()
def eval_rag(
    dataset_path: Annotated[
        str,
        typer.Argument(
            help="Path to the dataset to eval with. Can be an hf dataset dir, csv file, or path to hub file.",
            show_default=False,
        ),
    ],
    retriever_name_or_path: Annotated[
        str, typer.Option(help="Path to pretrained retriever or identifier from huggingface.co/models.")
    ],
    generator_name_or_path: Annotated[
        str, typer.Option(help="Path to pretrained (causal) generator or identifier from huggingface.co/models.")
    ],
    retriever_peft_model_path: Annotated[str, typer.Option(help="Path to the fine-tuned retriever peft layers")],
    generator_peft_model_path: Annotated[str, typer.Option(help="Path to the fine-tuned generator peft layers")],
    passage_column_name: Annotated[str, typer.Option(help="Name of the column containing the passage")] = "Abstract",
    query_column_name: Annotated[str, typer.Option(help="Name of the column containing the query")] = "Question",
    answer_column_name: Annotated[str, typer.Option(help="Name of the column containing the Answer")] = "Answer",
    embed_dim: Annotated[int, typer.Option(help="Dimension of the model embedding")] = 1024,
    max_length: Annotated[
        int, typer.Option(help="The max passage sequence length during tokenization. Longer sequences are truncated")
    ] = 128,
    test_batch_size: Annotated[int, typer.Option(help="Batch size (per device) for the test dataloader.")] = 8,
    query_batch_size: Annotated[int, typer.Option(help="Batch size (per device) for generator input")] = 16,
    device: Annotated[str, typer.Option(help="Device. cpu or cuda.")] = "cuda",
    torch_dtype: Annotated[
        TorchDtype, typer.Option(help="torch.dtype to use for tensors. float16 or bfloat16.")
    ] = TorchDtype.float16,
    top_k: Annotated[int, typer.Option(help="Top K retrieval")] = 10,
    evaluate_generator: Annotated[
        bool, typer.Option(help="Enable generator evaluation. If false, equivalent to eval-retriever")
    ] = True,
) -> None:
    """Evaluate your end-to-end rag generator and retriever"""
    evaluate_rag(
        dataset_or_path=dataset_path,
        retriever_name_or_path=retriever_name_or_path,
        generator_name_or_path=generator_name_or_path,
        retriever_peft_model_path=retriever_peft_model_path,
        generator_peft_model_path=generator_peft_model_path,
        passage_column_name=passage_column_name,
        query_column_name=query_column_name,
        answer_column_name=answer_column_name,
        embed_dim=embed_dim,
        max_length=max_length,
        test_batch_size=test_batch_size,
        query_batch_size=query_batch_size,
        device=device,
        torch_dtype=torch_dtype.value,
        # torch_dtype=cast(Literal["float16", "bfloat16"], torch_dtype.value),
        top_k=top_k,
        evaluate_generator=evaluate_generator,
    )


@cli.command()
def eval_retriever(
    dataset_path: Annotated[
        str,
        typer.Argument(
            help="Path to the dataset to eval with. Can be an hf dataset dir, csv file, or path to hub file.",
            show_default=False,
        ),
    ],
    retriever_name_or_path: Annotated[
        str, typer.Option(help="Path to pretrained retriever or identifier from huggingface.co/models.")
    ],
    retriever_peft_model_path: Annotated[str, typer.Option(help="Path to the fine-tuned retriever peft layers")],
    passage_column_name: Annotated[str, typer.Option(help="Name of the column containing the passage")] = "Abstract",
    query_column_name: Annotated[str, typer.Option(help="Name of the column containing the query")] = "Question",
    embed_dim: Annotated[int, typer.Option(help="Dimension of the model embedding")] = 1024,
    max_length: Annotated[
        int, typer.Option(help="The max passage sequence length during tokenization. Longer sequences are truncated")
    ] = 128,
    test_batch_size: Annotated[int, typer.Option(help="Batch size (per device) for the test dataloader.")] = 8,
    device: Annotated[str, typer.Option(help="Device. cpu or cuda.")] = "cuda",
    torch_dtype: Annotated[
        TorchDtype, typer.Option(help="torch.dtype to use for tensors. float16 or bfloat16.")
    ] = TorchDtype.float16,
    top_k: Annotated[int, typer.Option(help="Top K retrieval")] = 10,
) -> None:
    """Evaluate your retriever only"""
    evaluate_retriever(
        dataset_or_path=dataset_path,
        retriever_name_or_path=retriever_name_or_path,
        retriever_peft_model_path=retriever_peft_model_path,
        passage_column_name=passage_column_name,
        query_column_name=query_column_name,
        embed_dim=embed_dim,
        max_length=max_length,
        test_batch_size=test_batch_size,
        device=device,
        torch_dtype=torch_dtype.value,
        top_k=top_k,
    )


if __name__ == "__main__":
    cli()
