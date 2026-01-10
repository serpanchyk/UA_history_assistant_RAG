from src.ingest.ingesting import PDFIngestor

if __name__ == "__main__":
    ingestor = PDFIngestor()
    ingestor.run(filter_images_flag=True, link_images_flag=True)
