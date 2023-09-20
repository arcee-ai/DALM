import json

from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import MetadataMode
import pandas as pd

TRAIN_FILES = ["uber_2021.pdf"]
VAL_FILES = ["lyft_2021.pdf"]


def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SimpleNodeParser.from_defaults(chunk_size=512)
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes



train_nodes = load_corpus(TRAIN_FILES, verbose=True)
val_nodes = load_corpus(VAL_FILES, verbose=True)

train_df = pd.DataFrame({"text": [node.text for node in train_nodes], "title": [node.id_ for node in train_nodes]})
val_df = pd.DataFrame({"text": [node.text for node in val_nodes], "title": [node.id_ for node in val_nodes]})

train_df.to_csv("train_data.csv")
val_df.to_csv("val_data.csv")

