from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-2-7b-hf")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = field(default=False)
    reverse: bool = field(
        default=False,
        metadata={"help": "Reverse inst-resp for backward model $M_{yx}$ training"},
    )


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: str | None = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    final_lr: float = field(default=1e-6, metadata={"help": "Final minimal lr"})
