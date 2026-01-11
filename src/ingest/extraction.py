import pymupdf
import pandas as pd

from pathlib import Path
import uuid
from tqdm import tqdm

from src.fs_io.images import write_image
from src import IMAGES_DIR_PATH, TEXTBOOKS_DIR_PATH
from src.utils.normalize import normalize_text
from src.logger import logger

TEXT_BLOCK_TYPE = 0
IMAGE_BLOCK_TYPE = 1
MIN_TEXT_LENGTH = 5

def block_to_text(block: dict) -> str | None:
    """Convert PDF text block to string, return None if too short."""
    text = '\n'.join(
        ' '.join(span['text'] for span in line['spans'])
        for line in block['lines']
    )

    if len(text) < MIN_TEXT_LENGTH:
        return None

    return text

def extract_text_block(block: dict, doc_id: int, page_number: int) -> dict | None:
    """
    Extract and normalize a text block with metadata.
    Args:
        block (dict): PDF block data.
        doc_id (int): Document identifier.
        page_number (int): Page number in PDF.
    Returns:
        dict | None: Normalized text block with metadata or None if too short.
    """

    text = block_to_text(block)

    if text is None:
        return None

    block_info = {
        'text': normalize_text(text),
        'bbox': list(block['bbox']),
        'page': page_number,
        'doc_id': doc_id
    }

    return block_info

def save_image(image_bytes: bytes, ext: str, doc_id: int, page_number: int) -> str | None:
    """Save image bytes to disk, return path or None if no image."""
    if not image_bytes:
        logger.debug(f"No image found in doc {doc_id} page {page_number}")
        return None

    unique_id = uuid.uuid4().hex[: 8]
    image_path = IMAGES_DIR_PATH / f"doc{doc_id}_page{page_number}_{unique_id}.{ext}"
    write_image(image_bytes, image_path)

    return image_path

def extract_image(block: dict, doc_id: int, page_number: int) -> dict | None:
    """
    Extract an image from a PDF block and save it to disk.
    Args:
        block (dict): PDF block data.
        doc_id (int): Document identifier.
        page_number (int): Page number.
    Returns:
        dict | None: Image metadata with path and bbox or None if missing.
    """

    ext = block["ext"]
    image_bytes = block.get("image")

    image_path = save_image(image_bytes, ext, doc_id, page_number)
    if image_path is None:
        return None

    image_info = {
        'path': str(image_path),
        'bbox': list(block['bbox']),
        'page': page_number,
        'doc_id': doc_id
    }

    return image_info

def iter_pdf_pages(pdf_path: Path):
    """Yield pages from a PDF file."""
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            yield page

def iter_page_blocks(page: dict) -> list[dict]:
    """Return all blocks from a PDF page."""
    page_data = page.get_text("dict")
    return page_data.get("blocks")

def process_block(block: dict, doc_id: int, page_number: int) -> tuple[dict | None, dict | None]:
    """Return text and image info for a block, or (None, None)."""
    if block["type"] == TEXT_BLOCK_TYPE:
        return extract_text_block(block, doc_id, page_number), None

    elif block["type"] == IMAGE_BLOCK_TYPE:
        image_info = extract_image(block, doc_id, page_number)
        return extract_image(image_info, doc_id, page_number), image_info

    return None, None

def extract_blocks_from_pdf(pdf_path: Path, doc_id: int) -> tuple[list[dict], list[dict]]:
    """
    Extract all text and image blocks from a single PDF.
    Args:
        pdf_path (Path): Path to PDF file.
        doc_id (int): Document identifier.
    Returns:
        tuple[list[dict], list[dict]]: List of text blocks, list of image blocks.
    """

    texts: list[dict] = []
    images: list[dict] = []

    for page in iter_pdf_pages(pdf_path):
        for block in iter_page_blocks(page):
            text, image = process_block(block, doc_id, page.number)

            if text:
                texts.append(text)
            if image:
                images.append(image)

    return texts, images

def get_pdf_path(doc_row: tuple) -> Path | None:
    """Takes path of pdf from dataframe row and checks whether file exists."""
    pdf_path = TEXTBOOKS_DIR_PATH / doc_row.pdf_name

    if not pdf_path.exists():
        logger.error(f"PDF {pdf_path} does not exist")
        return None

    return pdf_path


def extract_data(df_docs: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """
    Extract text and image blocks from all PDFs listed in a DataFrame.
    Args:
        df_docs (pd.DataFrame): Must contain column 'pdf_name'.
    Returns:
        tuple[list[dict], list[dict]]: All text blocks and image blocks.
    """
    all_texts: list[dict] = []
    all_images: list[dict] = []

    for doc_row in tqdm(df_docs. itertuples(), total=len(df_docs), desc="Processing PDFs"):
        pdf_path = get_pdf_path(doc_row)
        if pdf_path is None:
            continue

        texts, images = extract_blocks_from_pdf(pdf_path, doc_row.Index)
        all_texts.extend(texts)
        all_images.extend(images)

    return all_texts, all_images