import pymupdf
import pandas as pd

from src.io.images import write_image
from src import IMAGES_DIR_PATH, TEXTBOOKS_DIR_PATH
from src.utils.normalize import normalize_text

TEXT_BLOCK = 0
IMAGE_BLOCK = 1

def extract_text_block(block: dict, doc_id: int, page_number: int) -> dict | None:

    text = '\n'.join(
        ' '.join(span['text'] for span in line['spans'])
        for line in block['lines']
    )

    # To not include noise text blocks
    if len(text) < 5:
        return None

    block_info = {
        'text': normalize_text(text),
        'bbox': tuple(block['bbox']),
        'page': page_number,
        'doc_id': doc_id
    }

    return block_info


def extract_image(block: dict, doc_id, page_number: int,  idx: int) -> dict | None:
    ext = block["ext"]
    image_bytes = block["image"]

    if not image_bytes:
        return None

    image_path = IMAGES_DIR_PATH / f"doc{doc_id}_page{page_number}_{idx}.{ext}"

    write_image(image_bytes, image_path)

    image_info = {
        'path': str(image_path),
        'bbox': tuple(block['bbox']),
        'page': page_number,
        'doc_id': doc_id
    }

    return image_info

def extract_data(df_docs: pd.DataFrame) -> tuple[list, list]:
    rows_text = []
    rows_images = []

    for doc_row in df_docs.itertuples():
        file_path = TEXTBOOKS_DIR_PATH / doc_row.pdf_name

        with pymupdf.open(file_path) as doc:
            for page in doc:

                page_data = page.get_text('dict')
                for i, block in enumerate(page_data['blocks']):

                    if block['type'] == TEXT_BLOCK:
                        text_info = extract_text_block(block, page.number, doc_row.doc_id)

                        if not text_info:
                            continue
                        rows_text.append(text_info)

                    if block['type'] == IMAGE_BLOCK:
                        image_info = extract_image(block, doc_row.doc_id, page.number, doc_row.doc_id)

                        if not image_info:
                            continue
                        rows_images.append(image_info)

    return rows_text, rows_images

