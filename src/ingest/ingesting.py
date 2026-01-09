import pandas as pd
from src.ingest.caption_linking import link_image_to_text
from src.ingest.extraction import extract_data
from src.ingest.filter_images import filter_images
from src.io.dataframes import read_parquet, write_parquet
from src.io.images import delete_images
from src import TEXTBOOKS_DF_PATH, TEXT_BLOCKS_DF_PATH, IMAGES_DF_PATH, IMAGES_DIR_PATH
from src.logger import logger


def run_ingesting() -> None:
    """
    Runs the full ingestion pipeline:
    1. Deletes old images.
    2. Reads textbooks DataFrame.
    3. Extracts text blocks and images from PDFs.
    4. Filters out invalid images.
    5. Links images to nearest text blocks.
    6. Saves processed text blocks and images as parquet files.
    """
    logger.info("Starting ingestion pipeline.")

    # Step 1: Clean previous images
    delete_images(IMAGES_DIR_PATH)
    logger.info(f"Deleted old images in {IMAGES_DIR_PATH}")

    # Step 2: Load textbooks DataFrame
    textbooks_df = read_parquet(TEXTBOOKS_DF_PATH)
    logger.info(f"Loaded textbooks DataFrame with {len(textbooks_df)} documents.")

    # Step 3: Extract text blocks and images from PDFs
    text_list, image_list = extract_data(textbooks_df)
    text_blocks_df = pd.DataFrame(text_list)
    images_df = pd.DataFrame(image_list)
    logger.info(f"Extracted {len(text_blocks_df)} text blocks and {len(images_df)} images.")

    # Step 4: Filter images
    images_df = filter_images(images_df)
    logger.info(f"{len(images_df)} images remain after filtering.")

    # Step 5: Link images to nearest text blocks
    images_df = link_image_to_text(images_df, text_blocks_df)
    logger.info("Linked images to text blocks.")

    # Step 6: Save results
    write_parquet(text_blocks_df, TEXT_BLOCKS_DF_PATH)
    write_parquet(images_df, IMAGES_DF_PATH)
    logger.info(f"Saved text blocks to {TEXT_BLOCKS_DF_PATH} and images to {IMAGES_DF_PATH}")

    logger.info("Ingestion pipeline finished successfully.")
