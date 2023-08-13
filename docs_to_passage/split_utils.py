
from typing import List, Optional

def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["Title"], documents["Abstract"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {"Title": titles, "Abstract": texts}


def filter_short_abstracts(example: dict) -> bool:
    abstract = example['Abstract']
    words = abstract.split()
    return len(words) >= 100