import pandas as pd

from src.ingest.caption_linking import link_image_to_text
from src.ingest.extraction import extract_data
from src.ingest.filter_images import filter_images
from src.io.dataframes import read_parquet, write_parquet

from src import TEXTBOOKS_DF_PATH, TEXT_BLOCKS_DF_PATH, IMAGES_DF_PATH, IMAGES_DIR_PATH
from src.io.images import delete_images


def run_ingesting():
    delete_images(IMAGES_DIR_PATH)

    textbooks_df = read_parquet(TEXTBOOKS_DF_PATH)

    text_blocks_df, images_df = tuple(
        pd.DataFrame(lst)
        for lst in extract_data(textbooks_df)
    )

    images_df = filter_images(images_df)

    images_df = link_image_to_text(images_df, text_blocks_df)

    write_parquet(text_blocks_df, TEXT_BLOCKS_DF_PATH)
    write_parquet(images_df, IMAGES_DF_PATH)

