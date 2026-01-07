import pymupdf
from pathlib import Path
import pandas as pd

import unicodedata
import re

from src import PROJECT_PATH

DOCS_PATH = PROJECT_PATH /  Path('data/pdfs')
IMAGES_PATH = PROJECT_PATH /  Path('data/images')
IMAGES_PATH.mkdir(parents=True, exist_ok=True)
DFS_PATH = PROJECT_PATH /  Path('data/dfs')

def normalise_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    text = text.replace("\u00A0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def scrap_text():
    """
    Takes texbook pdfs from docs dataset.
    Iterates through their pages. Retrieves text boxes and saves to dataset
    with box coordinates, page number and doc id. Saves text boxes to csv file.
    """

    df_docs = pd.read_csv(DFS_PATH / 'docs.csv')

    rows_text = []
    for index, doc_row in df_docs.iterrows():
        file_path = DOCS_PATH / doc_row['pdf_name']
        doc = pymupdf.open(file_path)

        for page in doc:
            page_data = page.get_text('dict')
            for block in page_data['blocks']:
                if block['type'] == 1:
                    continue
                text = ' '.join(
                    span['text']
                    for line in block['lines']
                    for span in line['spans']
                )

                rows_text.append({
                    'text': normalise_text(text),
                    'bbox': tuple(block['bbox']),
                    'page': page.number,
                    'doc_id': index
                })


    df_text = pd.DataFrame(rows_text)
    df_text.to_pickle(DFS_PATH / 'texts.pkl')

def scrap_images():
    """
    Takes texbook pdfs from docs dataset.
    Iterates through their pages. Retrieves images and saves to data dir
    and saves to dataset its path with box coordinates, page number and doc id.
    Saves dataset to csv file.
    """
    df_docs = pd.read_csv(DFS_PATH / 'docs.csv')

    rows_images = []
    for index, doc_row in df_docs.iterrows():
        file_path = DOCS_PATH / doc_row['pdf_name']
        doc = pymupdf.open(file_path)

        for page in doc:
            page_data = page.get_text('dict')
            for i, block in enumerate(page_data['blocks']):
                if block['type'] == 0:
                    continue

                ext = block["ext"]
                img_bytes = block["image"]

                image_path = IMAGES_PATH / f"doc{index}_page{page.number}_{i}.{ext}"

                with open(image_path, "wb") as f:
                    f.write(img_bytes)

                rows_images.append({
                    'image_path': str(image_path.name),
                    'bbox': tuple(block['bbox']),
                    'page': page.number,
                    'doc_id': index
                })

    df_images = pd.DataFrame(rows_images)
    df_images.to_pickle(DFS_PATH / 'images.pkl')