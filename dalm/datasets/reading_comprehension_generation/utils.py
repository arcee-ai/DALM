import csv
import os
import re
import tempfile
from typing import Dict, Iterator, List, Optional, Tuple

import sentencepiece as spm  # type: ignore[import-untyped]
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def input_generator(directory_or_file: str, csv_column: Optional[str] = None) -> Iterator[Tuple[str, str]]:
    """
    Generator that yields the file name and its contents
    """

    if os.path.isfile(directory_or_file) and not csv_column:
        raise ValueError("a CSV column must be specified if the input is a file")

    if os.path.isdir(directory_or_file):
        for file in os.listdir(directory_or_file):
            file_path = os.path.join(directory_or_file, file)
            if os.path.isfile(file_path):  # Ensures that we are reading files
                try:
                    with open(file_path, "r", encoding="utf-8") as file_contents:
                        contents = file_contents.read()
                except UnicodeDecodeError:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as file_contents:
                        contents = file_contents.read()

                yield file, contents

    elif os.path.isfile(directory_or_file) and directory_or_file.endswith(".csv") and csv_column:
        # If it's a CSV file, open it and yield the specified column
        with open(directory_or_file, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for index, row in enumerate(reader):
                yield os.path.basename(directory_or_file) + str(index), row[csv_column]
    else:
        raise ValueError("The input should be a directory or a CSV file.")


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
        spm.SentencePieceTrainer.train(
            input=text_file, model_prefix=model_prefix, vocab_size=32000, character_coverage=1.0
        )

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
