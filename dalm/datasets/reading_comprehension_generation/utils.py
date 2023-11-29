import csv
import os
import re
import tempfile
from typing import Dict, Iterator, List, Optional, Tuple

import sentencepiece as spm  # type: ignore[import-untyped]
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def input_generator(directory_or_file: str, csv_column: Optional[str] = None) -> Iterator[Tuple[str, str]]:
    """
    Generator that yields the contents of the files in the directory or the CSV column.
    """

    if os.path.isfile(directory_or_file):
        if directory_or_file.endswith(".csv") and csv_column:
            # Process a single CSV file
            yield from process_csv_file(directory_or_file, csv_column)
        elif not csv_column:
            # Process a single non-CSV file
            yield from process_plain_file(directory_or_file)
        else:
            raise ValueError("CSV column specified for non-CSV file")

    elif os.path.isdir(directory_or_file):
        # Process each file in the directory
        for file in os.listdir(directory_or_file):
            file_path = os.path.join(directory_or_file, file)
            if file_path.endswith(".csv") and csv_column:
                yield from process_csv_file(file_path, csv_column)
            elif not file_path.endswith(".csv"):
                yield from process_plain_file(file_path)

    else:
        raise ValueError("The input should be a directory or a file.")


def process_csv_file(file_path: str, csv_column: str) -> Iterator[Tuple[str, str]]:
    """Process a single CSV file."""
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for index, row in enumerate(reader):
            yield os.path.basename(file_path) + str(index), row[csv_column]


def process_plain_file(file_path: str) -> Iterator[Tuple[str, str]]:
    """Process a single plain text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file_contents:
            contents = file_contents.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="utf-8", errors="replace") as file_contents:
            contents = file_contents.read()
    yield os.path.basename(file_path), contents


def text_chunker(text: str, tokenizer: PreTrainedTokenizerBase, chunk_size: int) -> Iterator[str]:
    tokens = tokenizer(text, return_tensors="pt")["input_ids"]
    for i in range(0, tokens.shape[1], chunk_size):
        chunk = tokens[:, i : i + chunk_size]
        chunk = tokenizer.decode(chunk[0], skip_special_tokens=True)
        yield chunk


# standalone
def files_chunker(input_directory: str, model: str, context_length: int, output_directory: str, prompt: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model)

    tokens = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    CONSTANT = len(tokenizer(tokens)["input_ids"])

    k = context_length - CONSTANT

    for filename, text in input_generator(input_directory):
        extension = filename.split(".")[-1]
        output_file_name = filename.split(".")[0]
        for index, chunk in enumerate(text_chunker(text, tokenizer, k)):
            output_file = f"{output_file_name}_{index}.{extension}"
            with open(os.path.join(output_directory, output_file), "w") as o:
                o.write(chunk)


def create_domain_tokenizer(text_file: str) -> spm.SentencePieceProcessor:
    """
    train and return domain tokenizer
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define the model prefix with the path to the temp directory
        model_prefix = f"{temp_dir}/domain"

        # Train the SentencePiece model, the model is saved in the temporary directory
        exit_tries = False
        vocab_size = 32000
        # hack to get around that the vocab size is too large and has to be set manually
        # TODO: please figure out if mathematically a lower number of vocab is suitable for our case
        while not exit_tries:
            try:
                spm.SentencePieceTrainer.train(
                    input=text_file, model_prefix=model_prefix, vocab_size=vocab_size, character_coverage=1.0
                )
                exit_tries = True
            except RuntimeError as e:
                error_message = str(e)
                if error_message.startswith("Internal: src/trainer_interface.cc(661)"):
                    print(f"Vocab size of {vocab_size} is too large, decreasing it ...")
                    vocab_size = int(
                        error_message.split()[-1][:-1]
                    )  # error message ends with the recommended vocab and a period
                    print(f"Attempting with vocab size of {vocab_size}")
                else:
                    raise e

        sp_model_file = f"{model_prefix}.model"
        return spm.SentencePieceProcessor(model_file=sp_model_file)


def split_to_sentences(text: str) -> List[str]:
    sentences = re.split(r"[.?!]\s+", text)

    return sentences


def create_domain_tokenizer_from_files(directory_or_file: str, csv_column: Optional[str]) -> spm.SentencePieceProcessor:
    # open a tempfile and add sentences from files in directory_with_files to it
    with tempfile.TemporaryDirectory() as temp_dir:
        with open(os.path.join(temp_dir, "temp.txt"), "w", encoding="utf-8") as tfile:
            generator = input_generator(directory_or_file, csv_column)
            for _, text in generator:
                sentences = split_to_sentences(text)

            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence != "":
                    tfile.write(sentence + "\n")

        return create_domain_tokenizer(os.path.join(temp_dir, "temp.txt"))


def fix_first_prompt(text: str, chat_chain: List[Dict[str, str]]) -> List[Dict[str, str]]:
    # remove the first prompt
    first_prompt = chat_chain.pop(0)
    fixed_first_prompt = [
        {
            "content": f"Based on the following text: \n {text}, \n I'd like you to answer a few questions\n"
            + first_prompt["content"],
            "role": "user",
        }
    ]
    return fixed_first_prompt + chat_chain


# TODO: add test
# TODO: refactor this as a state machine?
def question_and_answer_extractor(whole_text: str, context: str) -> List[Dict[str, str]] | None:
    text_lines = whole_text.split("\n")
    question: List[str] = []
    answer: List[str] = []

    question_context = False
    answer_context = False

    result = []
    task_regex = r"^\*?\*?task\s*\d*"

    # question regex
    question_regex = r"^question\s*\d*"

    # answer regex
    answer_regex = r"^answer\s*\d*"

    for i in text_lines:
        raw_text = i.strip()
        text = raw_text.lower()

        # ignore empty lines
        if text == "":
            continue

        # if the line start matches the question regex or the task regex
        if re.match(question_regex, text) or re.match(task_regex, text):
            if answer_context:
                result.append({"content": " ".join(question), "role": "user"})
                result.append({"content": " ".join(answer), "role": "assistant"})
                question = []
                answer = []
                answer_context = False

            question_context = True
            answer_context = False

        if re.match(answer_regex, text):
            question_context = False
            answer_context = True

        if question_context:
            # remove (labelled as QUESTION and ANSWER) from the text
            raw_text = re.sub(r"\(labelled as QUESTION and ANSWER\)", "", raw_text)
            question.append(raw_text)

        if answer_context:
            answer.append(raw_text)

    if result == []:
        return None

    return fix_first_prompt(context, result)
