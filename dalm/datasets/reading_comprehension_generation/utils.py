import csv
import logging
import os
import re
import tempfile
from typing import Dict, Iterator, List, Optional, Tuple

import sentencepiece as spm  # type: ignore[import-untyped]
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


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
                    logger.warning(f"Vocab size of {vocab_size} is too large, decreasing it ...")
                    vocab_size = int(
                        error_message.split()[-1][:-1]
                    )  # error message ends with the recommended vocab and a period
                    logger.warning(f"Attempting with vocab size of {vocab_size}")
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


def wrap_context_with_rag_instruction(context: str) -> str:
    return f"Based on the following text: \n {context}, \n I'd like you to answer a few questions\n"


def extract_question(text: str) -> Tuple[bool, str]:
    """
    Extracts a question from a line of text.
    Returns a tuple of (is_question, question_text)
    """
    # question regex
    return extract_question_or_answer(text, extract_type="question")

def extract_answer(text: str) -> Tuple[bool, str]:
    """
    Extracts an answer from a line of text.
    Returns a tuple of (is_answer, answer_text)
    """
    # question regex
    return extract_question_or_answer(text, extract_type="answer")

def extract_question_or_answer(text: str, extract_type: str = "question") -> Tuple[bool, str]:

    # Match a line that starts with any number of junk characters, followed by either "question:" 
    # or "answer:", followed by any number of spaces (ignored), followed by any number of characters
    # that will be captured in a group as the question or answer.
    # extraction_regex = rf".*{extract_type}:\s*(.*)"

    # Update above to handle the case where the question or answer is in brackets, with
    # other text to be ignored inside the brackets
    extraction_regex = rf".*\[?{extract_type}[:\]]*(?:.*?\])?\s*(.*)"

    match = re.match(extraction_regex, text, re.IGNORECASE)
    extracted_text = match.group(1) if match else None
    found_extracted = True if extracted_text else False
    return found_extracted, extracted_text
    

def _raw_question_and_answer_extractor(whole_text: str) -> List[Dict[str, str]] | None:
    """
    Extracts questions and answers from the raw text generated by the large language model.

    @param whole_text: the raw questions and answers generated by the large language model, eg: 
                        "1. QUESTION: Can you summarize the .. ? 
                        ANSWER: Population imaging studies generated .."
    
    Algorithm overview:

    1. Loop over all lines in the text.  
    2. When we find a question, capture the question into a variable and set a state flag
    3. When we find an answer, capture the answer into a variable and save the QA pair
    4. When we run out of lines, return the list of QA pairs

    Supported formats are documented in the unit tests for this function: 
        tests/datasets/reading_comprehension_generation/test_utils.py

    Unsupported formats:

        Format 1: A question where the full question is in brackets

        [QUESTION: How can machine learning used in radiation oncology?]
        
        Format 2: A question which does not contain the keyword "Question":

        2. [type: true/false] Is the following sentence true? ... clinical trials.    
    """

    task_regex = r"^\*?\*?task\s*\d*"

    cur_qa_pair = {}
    qa_pairs = []

    state_waiting_for_question = "waiting_for_question"
    state_waiting_for_answer = "waiting_for_answer"
    state = state_waiting_for_question

    text_lines = whole_text.split("\n")
    for i in text_lines:
        raw_text = i.strip()
        text = raw_text.lower()

        # ignore empty lines
        if text == "":
            continue

        # If the line matches the task regex, print a warning.  The old code handled
        # "tasks", but this new code does not.  Need to inspect where these come into play
        if re.match(task_regex, text):
            logger.warning(f"Found a task line: {text}")

        if state == state_waiting_for_question:
            is_question, question_text = extract_question(text)
            if is_question:
                state = state_waiting_for_answer
                cur_qa_pair = {"question": question_text, "answer": "TBD"}
                continue
        elif state == state_waiting_for_answer:
            is_answer, answer_text = extract_answer(text)
            if is_answer:
                state = state_waiting_for_question
                cur_qa_pair["answer"] = answer_text
                if not cur_qa_pair["question"] or not cur_qa_pair["answer"]:
                    logger.warning(f"Found a QA pair with an empty question or answer: {cur_qa_pair}.  Skipping.")
                else:
                    qa_pairs.append(cur_qa_pair)
            else:
                # If we're expecting an answer, but the next non-empty line is not an answer,
                # something probably went wrong.  Print a warning and skip this QA pair.
                logger.warning(f"Found a question with no answer: {cur_qa_pair}.  Skipping.")
                state = state_waiting_for_question
            
            continue
        else:
            raise ValueError(f"Unknown state while extracting Q&A pairs: {state}")   

    return qa_pairs 


def convert_qa_pairs_to_chat_completions(qa_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Convert a list of QA pairs into a list of chat completions that can be fed into the large language model.
    """
    chat_completions = []
    for qa_pair in qa_pairs:
        question = qa_pair["question"]
        answer = qa_pair["answer"]

        question_chat_completion = {
            "content": question,
            "role": "user",
        }

        answer_chat_completion = {
            "content": answer,
            "role": "assistant",
        }

        chat_completions.append(question_chat_completion)
        chat_completions.append(answer_chat_completion)

    return chat_completions

def question_and_answer_extractor(whole_text: str, context: str) -> List[Dict[str, str]] | None:
    """

    Extracts questions and answers from the raw text generated by the large language model.

    @param whole_text: the raw questions and answers generated by the large language model, eg: 
                        "1. QUESTION: Can you summarize the .. ? 
                        ANSWER: Population imaging studies generated .."
    @param context: the full dataset text that was used to generate the questions and answers, eg:
                        "Population imaging studies generate data for developing and implementing..." 

    """

    chat_completion_inputs = []

    # Wrap the context with a RAG instruction 
    context_instruction = wrap_context_with_rag_instruction(context)

    # The first chat completion input is the context instruction
    first_chat_completion_input = {
        "content": context_instruction,
        "role": "user",
    }
    chat_completion_inputs.append(first_chat_completion_input)

    # Extract the qa pairs from whole_text
    qa_pairs = _raw_question_and_answer_extractor(whole_text)

    # If there are no qa pairs, return None
    if not qa_pairs:
        logger.warning(f"No QA pairs could be generated from whole_text: {whole_text} \n\n and context: {context}")
        return None

    # Convert the qa pairs to chat completion inputs
    qa_pairs_chat_completions = convert_qa_pairs_to_chat_completions(qa_pairs)

    # Add the qa pairs chat completions to the result
    chat_completion_inputs.extend(qa_pairs_chat_completions)

    return chat_completion_inputs