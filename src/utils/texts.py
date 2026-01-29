import re
import unicodedata
from typing import Any

import numpy as np

from src import TEXTBOOKS_DF_PATH
from src.fs_io.dataframes import read_parquet

MIN_TEXT_LENGTH = 10

ALLOWED_CATEGORIES = {"L", "N", "P", "Z"}

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove soft hyphen
    text = text.replace("\u00ad", "")

    # Fix hyphenated line breaks
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)

    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] == "C" or cat == "Cf":
            cleaned.append(" ")
        elif cat[0] in ALLOWED_CATEGORIES:
            cleaned.append(ch)
        # everything else is dropped

    text = "".join(cleaned)

    # Normalize spaces
    text = text.replace("\u00A0", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def block_to_text(block: dict) -> str | None:
    """Convert PDF text block to string, return None if too short."""
    text = '\n'.join(
        ' '.join(span['text']
        for span in line['spans'])
        for line in block['lines'])

    text = text

    normalized = normalize_text(text)

    if len(normalized) < MIN_TEXT_LENGTH:
        return None

    if not any(c.isalpha() for c in normalized):
        return None

    return normalized


def list_to_interval(lst: list | Any) -> str | None:
    """Converts list of integers to text: [1, 2, 3] -> '1-3' """
    if not isinstance(lst, list):
        lst = [lst]

    try:
        lst = [int(i) for i in lst]
    except ValueError:
        return None

    if len(lst) == 0:
        return ''

    max_i = max(lst)
    min_i = min(lst)

    if min_i == max_i:
        return str(min_i)

    return f'{min_i}-{max_i}'


def get_textbook_source(idx: int) -> str | None:
    """Get textbook source from its id. Returns None if it's invalid."""
    textbooks_df = read_parquet(TEXTBOOKS_DF_PATH)

    if idx in textbooks_df.index:
        return textbooks_df.loc[idx, 'source']
    return None

def chat_to_string(chat: list) -> str:
    return '\n'.join([f'{msg.type}: {msg.content}' for msg in chat])


def sanitize(obj):
    """Cast np objects to native python"""
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(sanitize(k)): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(i) for i in obj]
    return obj


