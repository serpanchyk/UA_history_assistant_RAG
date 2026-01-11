import re
import unicodedata

MIN_TEXT_LENGTH = 6

import re
import unicodedata

ALLOWED_CATEGORIES = {"L", "N", "P", "Z"}

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove soft hyphen
    text = text.replace("\u00ad", "")

    # Fix hyphenated line breaks
    text = re.sub(r"-\s*\n\s*", "", text)

    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] == "C":
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
    text = ' '.join(
        span['text']
        for line in block['lines']
        for span in line['spans']
    )

    normalized = normalize_text(text)

    if len(normalized) < MIN_TEXT_LENGTH:
        return None

    return normalized

