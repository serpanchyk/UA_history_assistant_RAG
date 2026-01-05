import pymupdf
from pathlib import Path
import pandas as pd

def scrap_data():
    docs_path = Path('../data/pdfs')
    images_path = Path('../data/images')
    dfs_path = Path('../data/dfs')

    df_docs = pd.read_csv(dfs_path / 'docs.csv', index_col='id')

    rows_text = []
    rows_images = []
    for index, doc_row in df_docs.iterrows():
        file_path = docs_path / doc_row['pdf_name']
        doc = pymupdf.open(file_path)

        for page in doc:
            page_data = page.get_text('dict')
            for block in page_data['blocks']:
                if block['type'] == 0:
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

                elif block['type'] == 1:
                    image_path = images_path / f"doc{index}_page{page.number}_{block['number']}.{block['ext']}"
                    with open(image_path, "wb") as f:
                        f.write(block['image'])

                    rows_images.append({
                        'image_path': str(image_path.name),
                        'bbox': tuple(block['bbox']),
                        'page': page.number,
                        'doc_id': index
                    })

    df_text = pd.DataFrame(rows_text)
    df_images = pd.DataFrame(rows_images)

    df_text.to_csv(dfs_path / 'texts.csv')
    df_images.to_csv(dfs_path / 'images.csv')


if __name__ == '__main__':
    scrap_data()