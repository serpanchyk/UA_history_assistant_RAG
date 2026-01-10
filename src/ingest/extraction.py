import pymupdf
import pandas as pd

from pathlib import Path
import uuid

from src.io.images import write_image
from src import IMAGES_DIR_PATH, TEXTBOOKS_DIR_PATH
from src.utils.normalize import normalize_text
from src.logger import logger

TEXT_BLOCK_TYPE = 0
IMAGE_BLOCK_TYPE = 1
MIN_TEXT_LENGTH = 5


def extract_text_block(block: dict, doc_id: int, page_number: int) -> dict | None:
    """
    Extracts a text block from a PDF page.
    Args:
        block (dict): Block data from pymupdf page.
        doc_id (int): Document identifier.
        page_number (int): Page number in the PDF.
    Returns:
        dict | None: Normalized text block with metadata, or None if block is too short.
    """
    text = '\n'.join(
        ' '.join(span['text'] for span in line['spans'])
        for line in block['lines']
    )

    if len(text) < MIN_TEXT_LENGTH:
        logger.debug(f"Skipping short text block on doc {doc_id} page {page_number}")
        return None

    block_info = {
        'text': normalize_text(text),
        'bbox': list(block['bbox']),
        'page': page_number,
        'doc_id': doc_id
    }

    return block_info


def extract_image(block: dict, doc_id: int, page_number: int) -> dict | None:
    """
    Extracts an image from a PDF page and saves it to disk.
    Args:
        block (dict): Block data from pymupdf page.
        doc_id (int): Document identifier.
        page_number (int): Page number.
        idx (int): Index of the block in the page.
    Returns:
        dict | None: Image metadata with path and bbox, or None if image is missing.
    """
    ext = block["ext"]
    image_bytes = block.get("image")

    if not image_bytes:
        logger.debug(f"No image found in doc {doc_id} page {page_number}")
        return None

    unique_id = uuid.uuid4().hex[: 8]
    image_path = IMAGES_DIR_PATH / f"doc{doc_id}_page{page_number}_{unique_id}.{ext}"
    write_image(image_bytes, image_path)

    image_info = {
        'path': str(image_path),
        'bbox': list(block['bbox']),
        'page': page_number,
        'doc_id': doc_id
    }

    return image_info


def extract_blocks_from_pdf(pdf_path: Path, doc_id: int) -> tuple[list[dict], list[dict]]:
    """
    Extracts all text blocks and images from a single PDF.
    Args:
        pdf_path (Path): Path to the PDF file.
        doc_id (int): Document identifier.
    Returns:
        tuple[list[dict], list[dict]]: List of text blocks, list of image blocks.
    """

    texts: list[dict] = []
    images: list[dict] = []

    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            page_data = page.get_text("dict")
            for block in page_data["blocks"]:
                if block["type"] == TEXT_BLOCK_TYPE:
                    text_info = extract_text_block(block, doc_id, page.number)
                    if text_info:
                        texts.append(text_info)
                elif block["type"] == IMAGE_BLOCK_TYPE:
                    image_info = extract_image(block, doc_id, page.number)
                    if image_info:
                        images.append(image_info)

    return texts, images


def extract_data(df_docs: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """
    Extracts text blocks and images from all PDFs listed in the DataFrame.
    Args:
        df_docs (pd.DataFrame): Must contain 'pdf_name'.
    Returns:
        tuple[list[dict], list[dict]]: All text blocks and images.
    """
    all_texts: list[dict] = []
    all_images: list[dict] = []

    for doc_row in df_docs.itertuples():
        pdf_path = TEXTBOOKS_DIR_PATH / doc_row.pdf_name
        if not pdf_path.exists():
            logger.error(f"PDF {pdf_path} does not exist")
            continue

        texts, images = extract_blocks_from_pdf(pdf_path, doc_row.Index)
        all_texts.extend(texts)
        all_images.extend(images)

    logger.info(f"Extracted {len(all_texts)} text blocks and {len(all_images)} images from corpus")
    return all_texts, all_images

