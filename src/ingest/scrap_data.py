import pymupdf
import pandas as pd

from src import IMAGES_DF_PATH, TEXTBOOKS_DF_PATH, IMAGES_DIR_PATH, TEXTBOOKS_DIR_PATH, \
    TEXT_BLOKS_DF_PATH
from src.utils.normalize import normalize_text

TEXT_BLOCK = 0
IMAGE_BLOCK = 1

def extract_text_blocks():
    """
    Takes texbook pdfs from docs dataset.
    Iterates through their pages. Retrieves text boxes and saves to dataset
    with box coordinates, page number and doc id. Saves text boxes to csv file.
    """

    df_docs = pd.read_csv(TEXTBOOKS_DF_PATH)

    for doc_id, doc_row in df_docs.iterrows():
        file_path = TEXTBOOKS_DIR_PATH / doc_row['name']

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

    df_text = pd.DataFrame(rows_text)
    df_text.to_pickle(TEXT_BLOKS_DF_PATH)

def extract_images():
    """
    Takes texbook pdfs from docs dataset.
    Iterates through their pages. Retrieves images and saves to data dir
    and saves to dataset its path with box coordinates, page number and doc id.
    Saves dataset to csv file.
    """
    df_docs = pd.read_csv(TEXTBOOKS_DF_PATH)

    for doc_id, doc_row in df_docs.iterrows():
        file_path = TEXTBOOKS_DIR_PATH / doc_row['name']

        with pymupdf.open(file_path) as doc:
            rows_images = []
            for page in doc:
                page_data = page.get_text('dict')
                for i, block in enumerate(page_data['blocks']):
                    if block['type'] == TEXT_BLOCK:
                        continue

                    ext = block["ext"]
                    img_bytes = block["image"]

                    if not img_bytes:
                        continue

                    image_path = IMAGES_DIR_PATH / f"doc{doc_id}_page{page.number}_{i}.{ext}"

                    with open(image_path, "wb") as f:
                        f.write(img_bytes)

                    rows_images.append({
                        'image_path': str(image_path.name),
                        'bbox': tuple(block['bbox']),
                        'page': page.number,
                        'doc_id': doc_id
                    })

    df_images = pd.DataFrame(rows_images)
    df_images.to_pickle(IMAGES_DF_PATH)