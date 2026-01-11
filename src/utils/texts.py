import re
import unicodedata

MIN_TEXT_LENGTH = 6

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
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

