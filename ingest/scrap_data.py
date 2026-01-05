import pymupdf
from pathlib import Path
import pandas as pd

DOCS_PATH = Path('../data/pdfs')
IMAGES_PATH = Path('../data/images')
IMAGES_PATH.mkdir(parents=True, exist_ok=True)
DFS_PATH = Path('../data/dfs')

def scrap_text():
    df_docs = pd.read_csv(DFS_PATH / 'docs.csv', index_col='id')

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
                    'text': text,
                    'bbox': tuple(block['bbox']),
                    'page': page.number,
                    'doc_id': index
                })


    df_text = pd.DataFrame(rows_text)

    df_text.to_csv(DFS_PATH / 'texts.csv')

def scrap_images():
    df_docs = pd.read_csv(DFS_PATH / 'docs.csv', index_col='id')

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
    df_images.to_csv(DFS_PATH / 'images.csv')


if __name__ == '__main__':
    scrap_text()
    scrap_images()