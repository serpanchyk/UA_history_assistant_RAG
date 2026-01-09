import pymupdf
import pandas as pd

from src.io.images import write_image
from src import IMAGES_DF_PATH, TEXTBOOKS_DF_PATH, IMAGES_DIR_PATH, TEXTBOOKS_DIR_PATH
from src.utils.normalize import normalize_text

TEXT_BLOCK = 0
IMAGE_BLOCK = 1

def extract_text_blocks(df_docs: pd.DataFrame) -> list[dict]:

    for doc_id, doc_row in df_docs.iterrows():
        file_path = TEXTBOOKS_DIR_PATH / doc_row['pdf_name']

        with pymupdf.open(file_path) as doc:
            rows_text = []
            for page in doc:
                page_data = page.get_text('dict')
                for block in page_data['blocks']:
                    if block['type'] == IMAGE_BLOCK:
                        continue

                    text = '\n'.join(
                        ' '.join(span['text'] for span in line['spans'])
                        for line in block['lines']
                    )

                    # To not include noise text blocks
                    if len(text) < 5:
                        continue

                    rows_text.append({
                        'text': normalize_text(text),
                        'bbox': tuple(block['bbox']),
                        'page': page.number,
                        'doc_id': doc_id
                    })

    return rows_text

def extract_images(df_docs: pd.DataFrame) -> list[dict]:

    for doc_id, doc_row in df_docs.iterrows():
        file_path = TEXTBOOKS_DIR_PATH / doc_row['pdf_name']

        with pymupdf.open(file_path) as doc:
            rows_images = []
            for page in doc:
                page_data = page.get_text('dict')
                for i, block in enumerate(page_data['blocks']):
                    if block['type'] == TEXT_BLOCK:
                        continue

                    ext = block["ext"]
                    image_bytes = block["image"]

                    if not image_bytes:
                        continue

                    image_path = IMAGES_DIR_PATH / f"doc{doc_id}_page{page.number}_{i}.{ext}"

                    write_image(image_bytes, image_path)

                    rows_images.append({
                        'image_path': str(image_path.name),
                        'bbox': tuple(block['bbox']),
                        'page': page.number,
                        'doc_id': doc_id
                    })

    return rows_images