# utils
from collections import defaultdict
from string import punctuation

# typing
from typing import List, Dict


def clean_text(text: str) -> str:
    """Cleans the text by removing punctuation and converting to lowercase.

    Note that it removes hyphens and underscores, replace them with spaces
    before applying this function if that is desired.

    Parameters
    ----------
    text : str
        The text to be cleaned.

    Returns
    -------
    str
        The cleaned text.
    """

    return text.translate(str.maketrans("", "", punctuation)).lower().strip()


def find_stopwords(docs: List[str], threshold: float = 0.6) -> List[str]:
    """Find stopwords in the given documents.

    Parameters
    ----------
    docs : List[str]
        List of documents.
    threshold : float, optional
        Threshold for word frequency, by default 0.6

    Returns
    -------
    List[str]
        List of stopwords.
    """
    stopwords: List[str] = list()
    frequency: Dict[str, int] = defaultdict(int)

    for doc in docs:
        for word in doc.split():
            frequency[word] += 1

    for word, freq in frequency.items():
        if freq / len(docs) > threshold:
            stopwords.append(word)

    return stopwords
