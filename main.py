from src.ingest.scrap_data import scrap_text, scrap_images
from src.ingest.filter_images import filter_images

if __name__ == '__main__':
    scrap_images()
    filter_images()