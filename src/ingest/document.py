import pymupdf

from src import TEXTBOOKS_DIR_PATH, IMAGES_DIR_PATH
from src.utils.normalize import normalize_text


class Document:
    def __init__(self, id, name):
        self.id = id
        self.name = name

    def extract_text_blocks(self):
        file_path = TEXTBOOKS_DIR_PATH / self.name
        doc = pymupdf.open(file_path)

        rows_text = []
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
                    'text': normalize_text(text),
                    'bbox': tuple(block['bbox']),
                    'page': page.number,
                    'doc_id': self.id
                })

    def extract_images(self):

        file_path = TEXTBOOKS_DIR_PATH / self.name
        doc = pymupdf.open(file_path)

        rows_images = []
        for page in doc:
            page_data = page.get_text('dict')
            for i, block in enumerate(page_data['blocks']):
                if block['type'] == 0:
                    continue

                ext = block["ext"]
                img_bytes = block["image"]

                image_path = IMAGES_DIR_PATH / f"doc{id}_page{page.number}_{i}.{ext}"

                with open(image_path, "wb") as f:
                    f.write(img_bytes)

                rows_images.append({
                    'image_path': str(image_path.name),
                    'bbox': tuple(block['bbox']),
                    'page': page.number,
                    'doc_id': self.id
                })