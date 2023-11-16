import os
import tempfile
import re

from transformers import AutoTokenizer


def list_dir(directory):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        with open(file_path, "r") as file_contents:
            contents = file_contents.read()
        yield file, contents


def text_chunker(text, tokenizer, chunk_size):
    tokens = tokenizer(text, return_tensors="pt")["input_ids"]
    for i in range(0, tokens.shape[1], chunk_size):
        chunk = tokens[:, i : i + chunk_size]
        chunk = tokenizer.decode(chunk[0], skip_special_tokens=True)
        yield chunk


# standalone
def files_chunker(input_directory, model, context_length, output_directory, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model)

    tokens = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    CONSTANT = len(tokenizer(tokens)["input_ids"])

    k = context_length - CONSTANT

    for filename, text in list_dir(input_directory):
        extension = filename.split(".")[-1]
        output_file_name = filename.split(".")[0]
        for index, chunk in enumerate(text_chunker(text, tokenizer, k)):
            output_file = f"{output_file_name}_{index}.{extension}"
            with open(os.path.join(output_directory, output_file), "w") as o:
                o.write(chunk)


def create_domain_tokenizer(text):
    """
    train and return domain tokenizer
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define the model prefix with the path to the temp directory
        model_prefix = f"{temp_dir}/domain"

        # Train the SentencePiece model, the model is saved in the temporary directory
        spm.SentencePieceTrainer.train(input=text, model_prefix=model_prefix, vocab_size=32000, character_coverage=1.0)

        sp_model_file = f"{model_prefix}.model"
        return spm.SentencePieceProcessor(model_file=sp_model_file)


def fix_first_prompt(text, chat_chain):
    # remove the first prompt
    first_prompt = chat_chain.pop(0)
    first_prompt = [
        {
            "content": f"Based on the following text: \n {text}, \n I'd like you to answer a few questions\n"
            + first_prompt["content"],
            "role": "user",
        }
    ]
    return first_prompt + chat_chain


# TODO: type hinting is very necessary here
# TODO: add test
def question_and_answer_extractor(whole_text, context):
    whole_text = whole_text.split("\n")
    question = []
    answer = []

    question_context = False
    answer_context = False

    result = []
    task_regex = r"^\*?\*?task\s*\d*"

    # question regex
    question_regex = r"^question\s*\d*"

    # answer regex
    answer_regex = r"^answer\s*\d*"

    for i in whole_text:
        raw_text = i.strip()
        text = raw_text.lower()

        # ignore empty lines
        if text == "":
            continue

        # task regex to match Task 1, task 1 , task
        task_regex = r"^\*?\*?task\s*\d*"

        # question regex
        question_regex = r"^question\s*\d*"

        # answer regex
        answer_regex = r"^answer\s*\d*"

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
