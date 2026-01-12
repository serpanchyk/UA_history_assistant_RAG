from pathlib import Path
import pandas as pd
from src.ingest.caption_linking import link_image_to_text
from src.ingest.chunking import chunking
from src.ingest.extraction import extract_data
from src.ingest.filter_images import filter_images
from src.fs_io.dataframes import read_parquet, write_parquet
from src.fs_io.images import delete_images
from src import TEXTBOOKS_DF_PATH, TEXT_BLOCKS_DF_PATH, IMAGES_DF_PATH, IMAGES_DIR_PATH, CHUNKS_DF_PATH
from src.logger import logger


class PDFIngestor:
    """
    One-time PDF ingestion pipeline for a RAG project.

    Responsibilities:
    - Deletes old images
    - Loads textbooks metadata
    - Extracts text blocks and images from PDFs
    - Optionally filters images
    - Optionally links images to text blocks
    - Saves processed data to parquet files

    Note:
        This is intended for one-time ingestion from scratch.
        It deletes IMAGES_DIR_PATH before processing.
    """

    def __init__(self, images_dir: Path = IMAGES_DIR_PATH):
        self.images_dir: Path = images_dir

    def delete_old_images(self, force: bool = False) -> None:
        """Deletes all images in the images directory."""
        logger.info(f"Deleting old images in {self.images_dir}")
        delete_images(self.images_dir, force=force)
        logger.info("Old images deleted.")

    def load_textbooks(self) -> pd.DataFrame:
        """Loads textbooks metadata from parquet."""
        textbooks_df: pd.DataFrame = read_parquet(TEXTBOOKS_DF_PATH)
        logger.info(f"Loaded textbooks DataFrame with {len(textbooks_df)} documents.")
        return textbooks_df

    def extract_data_from_pdfs(self, textbooks_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extracts text blocks and images from PDFs.
        Args:
            textbooks_df (pd.DataFrame): DataFrame of textbooks.
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: text_blocks_df, images_df
        """
        text_list, image_list = extract_data(textbooks_df)
        text_blocks_df: pd.DataFrame = pd.DataFrame(text_list)
        images_df: pd.DataFrame = pd.DataFrame(image_list)

        logger.info(f"Extracted {len(text_blocks_df)} text blocks and {len(images_df)} images.")
        return text_blocks_df, images_df

    def filter_images_df(self, images_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out unwanted or invalid images.
        Args:
            images_df (pd.DataFrame): DataFrame containing images.
        Returns:
            pd.DataFrame: Filtered images DataFrame.
        """
        filtered_df: pd.DataFrame = filter_images(images_df)
        logger.info(f"{len(filtered_df)} images remain after filtering.")
        return filtered_df

    def link_images_to_text(self, images_df: pd.DataFrame, text_blocks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Links each image to the nearest text block.
        Args:
            images_df (pd.DataFrame): Images DataFrame.
            text_blocks_df (pd.DataFrame): Text blocks DataFrame.
        Returns:
            pd.DataFrame: Images DataFrame with 'caption' column.
        """
        linked_df: pd.DataFrame = link_image_to_text(images_df, text_blocks_df)
        logger.info("Linked images to text blocks.")
        return linked_df

    def chunking_df(self, text_blocks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggreates text blocks to bigger chunks.
        Args:
            text_blocks_df (pd.DataFrame): Text blocks DataFrame.
        Returns:
            pd.DataFrame: Chunks DataFrame.
        """
        chunks: list[dict] = chunking(text_blocks_df)
        chunks_df: pd.DataFrame = pd.DataFrame(chunks)
        logger.info("Converted text blocks to chunks.")
        return chunks_df

    def save_results(
            self,
            text_blocks_df: pd.DataFrame,
            images_df: pd.DataFrame,
            chunks_df: pd.DataFrame
    ) -> None:
        """
        Saves processed DataFrames to parquet files.
        Args:
            text_blocks_df (pd.DataFrame): Text blocks DataFrame.
            images_df (pd.DataFrame): Images DataFrame.
        """
        write_parquet(text_blocks_df, TEXT_BLOCKS_DF_PATH)
        write_parquet(images_df, IMAGES_DF_PATH)
        write_parquet(chunks_df, CHUNKS_DF_PATH)
        logger.info(
            f"Saved text blocks to {TEXT_BLOCKS_DF_PATH}, "
            f"images to {IMAGES_DF_PATH} "
            f"and chunks to {CHUNKS_DF_PATH}."
        )

    def run(
            self,
            filter_images_flag: bool = True,
            link_images_flag: bool = True,
    ) -> None:
        """
        Runs the full ingestion pipeline.
        Args:
            filter_images_flag (bool): If True, filters images.
            link_images_flag (bool): If True, links images to nearest text blocks.
        """

        logger.info("Starting PDF ingestion pipeline.")
        try:
            self.delete_old_images(force=True)
            textbooks_df = self.load_textbooks()
            text_blocks_df, images_df = self.extract_data_from_pdfs(textbooks_df)

            if filter_images_flag:
                images_df = self.filter_images_df(images_df)

            if link_images_flag:
                images_df = self.link_images_to_text(images_df, text_blocks_df)

            chunks_df = self.chunking_df(text_blocks_df)

            self.save_results(text_blocks_df, images_df, chunks_df)
            logger.info("PDF ingestion pipeline finished successfully.")
        except Exception as err:
            logger.error(f"Pipeline failed: {err}")
            raise