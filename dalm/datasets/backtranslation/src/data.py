import torch
from fastchat.conversation import Conversation, SeparatorStyle, get_conv_template
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.utils.constant import IGNORE_TOKEN_ID
from src.utils.io import load_jsonlines
from src.utils.print import rank0_print


def preprocess(
    sources,
    tokenizer: PreTrainedTokenizer,
    reverse: bool = False,
) -> dict:
    if reverse:
        conv = get_conv_template("vicuna_v1.1_reverse")
        aug_conv = conv
    else:
        conv = get_conv_template("vicuna_v1.1_seed")
        aug_conv = get_conv_template("vicuna_v1.1_aug")
    assert conv.roles == aug_conv.roles
    assert conv.sep == aug_conv.sep
    assert conv.sep2 == aug_conv.sep2

    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        conv_src = source["source"]
        if conv_src == "aug":
            _conv = aug_conv
        else:
            _conv = conv
        if roles[source["conversations"][0]["from"]] != _conv.roles[0]:
            # Skip the first one if it is not from human
            source["conversations"] = source["conversations"][1:]

        _conv.messages = []
        for j, sentence in enumerate(source["conversations"]):
            role = roles[sentence["from"]]
            assert role == _conv.roles[j % 2], f"{i}"
            _conv.append_message(role, sentence["value"])
        conversations.append(_conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def convert_inst_resp_pairs_into_fastchat(ins: dict, reverse: bool = False) -> dict:
    inst = ins["instruction"] if not reverse else ins["response"]
    resp = ins["response"] if not reverse else ins["instruction"]
    if "score" in ins:
        source = "aug"
    else:
        source = "seed"
    return {
        "id": "",
        "source": source,
        "conversations": [
            {"from": "human", "value": inst},
            {"from": "gpt", "value": resp},
        ],
    }


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        tokenizer: PreTrainedTokenizer,
        reverse: bool = False,
    ):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [
            convert_inst_resp_pairs_into_fastchat(example, reverse=reverse)
            for example in raw_data
        ]
        data_dict = preprocess(sources, tokenizer, reverse=reverse)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        tokenizer: PreTrainedTokenizer,
        reverse: bool = False,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.reverse = reverse

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess(
            [
                convert_inst_resp_pairs_into_fastchat(
                    self.raw_data[i], reverse=self.reverse
                )
            ],
            self.tokenizer,
            reverse=self.reverse,
        )
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(tokenizer: PreTrainedTokenizer, data_args) -> dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_data = load_jsonlines(data_args.data_path)
    train_dataset = dataset_cls(
        train_data, tokenizer=tokenizer, reverse=data_args.reverse
    )

    if data_args.eval_data_path:
        eval_data = load_jsonlines(data_args.eval_data_path)
        eval_dataset = dataset_cls(
            eval_data, tokenizer=tokenizer, reverse=data_args.reverse
        )
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


class InferenceDataset(Dataset):
    def __init__(
        self,
        data,
        content_name: str = "content",
        reverse: bool = False,
    ):
        self.data = data
        self.reverse = reverse
        self.content_name = content_name

        if reverse:
            self.conv: Conversation = get_conv_template("vicuna_v1.1_reverse")
        else:
            self.conv: Conversation = get_conv_template("vicuna_v1.1")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins = self.data[idx]
        self.conv.messages.clear()
        self.conv.append_message(self.conv.roles[0], ins[self.content_name])
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        return prompt

    def get_all(self):
        return [self[i] for i in range(len(self))]


class CollateFnWithTokenization:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_len: int = 2048) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        outputs = self.tokenizer(
            batch,
            return_tensors="pt",
            max_length=self.max_seq_len,
            padding=True,
            truncation=True,
        )
        return outputs
