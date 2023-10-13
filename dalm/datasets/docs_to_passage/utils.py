from typing import List

DEFAULT_MAX_WORDS = 100
DEFAULT_MIN_WORDS = 5
TITLE_COL = "title"
TEXT_COL = "text"

def make_sure_dalm_is_installed():
    print("dalm is installed")


def split_text(text: str, n: int = DEFAULT_MAX_WORDS, character: str = " ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    texts = text.split(character)
    return [character.join(texts[i : i + n]).strip() for i in range(0, len(texts), n)]


def split_documents(documents: dict, max_words: int = DEFAULT_MAX_WORDS) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents[TITLE_COL], documents[TEXT_COL], strict=True):
        if text is not None:
            for passage in split_text(text, n=max_words):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {TITLE_COL: titles, TEXT_COL: texts}


def keep_sufficiently_long_passages(example: dict, min_words: int = DEFAULT_MIN_WORDS) -> bool:
    """Returns True (for a filter) when the passage has more than min_words in it. Default 5 words"""
    abstract = example[TEXT_COL]
    words = abstract.split()
    return len(words) >= min_words
