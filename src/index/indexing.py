from src import CHUNKS_DF_PATH, IMAGES_DF_PATH
from src.fs_io.dataframes import read_parquet
from src.index.embedder import EmbeddingService
from src.index.storage import QdrantVectorStore
from src.logger import logger


def run_indexing():
    logger.info("Starting indexing process...")

    embedder = EmbeddingService()

    vector_store = QdrantVectorStore(embedding_service=embedder)

    if CHUNKS_DF_PATH.exists():
        try:
            logger.info(f"Loading chunks from {CHUNKS_DF_PATH}")
            chunks_df = read_parquet(CHUNKS_DF_PATH)

            text_chunks = chunks_df.to_dict(orient="records")
            vector_store.add_text_entry(text_chunks)

            logger.info("Text indexing complete.")
        except Exception as error:
            logger.exception("Text indexing failed", exc_info=error)

    else:
        logger.warning(f"Chunks dataframe {CHUNKS_DF_PATH} does not exist. Skipping text indexing.")

    if IMAGES_DF_PATH.exists():
        try:
            logger.info(f"Loading images from {IMAGES_DF_PATH}")
            images_df = read_parquet(IMAGES_DF_PATH)

            vector_store.add_image_entry(images_df)
            logger.info("Image indexing complete.")
        except Exception as error:
            logger.exception("Image indexing failed", exc_info=error)

    else:
        logger.warning(f"Images dataframe {IMAGES_DF_PATH} does not exist. Skipping image indexing.")

    logger.info("Finished indexing process.")

