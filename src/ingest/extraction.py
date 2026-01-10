import pymupdf
import pandas as pd

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


def extract_data(df_docs: pd.DataFrame) -> tuple[list, list]:
    """
    Extracts text blocks and images from all PDF documents in the DataFrame.
    Args:
        df_docs (pd.DataFrame): DataFrame with document info, must contain 'pdf_name'.
    Returns:
        tuple[list, list]: List of text block dicts, list of image dicts.
    """
    rows_text = []
    rows_images = []

    for doc_row in df_docs.itertuples():
        file_path = TEXTBOOKS_DIR_PATH / doc_row.pdf_name
        logger.info(f"Opening PDF: {file_path}")

        if not file_path.exists():
            logger.error(f"PDF {file_path} does not exist")
            continue
        with pymupdf.open(file_path) as doc:
            for page in doc:
                page_data = page.get_text('dict')
                for i, block in enumerate(page_data['blocks']):

                    if block['type'] == TEXT_BLOCK_TYPE:
                        text_info = extract_text_block(block, doc_row.Index, page.number)
                        if text_info:
                            rows_text.append(text_info)

                    elif block['type'] == IMAGE_BLOCK_TYPE:
                        image_info = extract_image(block, doc_row.Index, page.number, i)
                        if image_info:
                            rows_images.append(image_info)

    logger.info(f"Extracted {len(rows_text)} text blocks and {len(rows_images)} images")
    return rows_text, rows_images
