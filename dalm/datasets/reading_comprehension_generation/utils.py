import os
import tempfile

from transformers import AutoTokenizer

def list_dir(directory):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as file_contents:
            contents = file_contents.read()
        yield file, contents


def text_chunker(text, tokenizer, chunk_size):
    tokens = tokenizer(text, return_tensors='pt')['input_ids']
    for i in range(0, tokens.shape[1], chunk_size):
        chunk = tokens[:, i:i+chunk_size]
        chunk = tokenizer.decode(chunk[0], skip_special_tokens=True)
        yield chunk


# standalone
def files_chunker(input_directory, model, context_length, output_directory, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model)

    tokens = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    CONSTANT = len(tokenizer(tokens)["input_ids"])

    k = context_length - CONSTANT

    for filename, text in list_dir(input_directory):
        extension = filename.split('.')[-1]
        output_file_name = filename.split('.')[0]
        for index, chunk in enumerate(text_chunker(text, tokenizer, k)):
            output_file = f'{output_file_name}_{index}.{extension}'
            with open(os.path.join(output_directory, output_file), 'w') as o:
                o.write(chunk)


def create_domain_tokenizer(text):
    """
    train and return domain tokenizer
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define the model prefix with the path to the temp directory
        model_prefix = f"{temp_dir}/domain"

        # Train the SentencePiece model, the model is saved in the temporary directory
        spm.SentencePieceTrainer.train(
            input=text,
            model_prefix=model_prefix,
            vocab_size=32000,
            character_coverage=1.0
        )

        sp_model_file = f"{model_prefix}.model"
        return spm.SentencePieceProcessor(model_file=sp_model_file)