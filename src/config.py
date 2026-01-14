from functools import cached_property
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings.

    Values can be overridden by environment variables (e.g., RAG_DATA_DIR)
    or a .env file in the project root.
    """

    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    DATA_DIR: Path = PROJECT_ROOT / "data"

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        extra="ignore"
    )

    @computed_field
    @cached_property
    def DFS_PATH(self) -> Path:
        return self.DATA_DIR / "dfs"

    @computed_field
    @cached_property
    def IMAGES_DIR_PATH(self) -> Path:
        return self.DATA_DIR / "images"

    @computed_field
    @cached_property
    def TEXTBOOKS_DIR_PATH(self) -> Path:
        return self.DATA_DIR / "pdfs"

    @computed_field
    @cached_property
    def REJECTED_IMAGES_DIR_PATH(self) -> Path:
        return self.IMAGES_DIR_PATH / "rejected_images"

    @computed_field
    @cached_property
    def TEXTBOOKS_DF_PATH(self) -> Path:
        return self.DFS_PATH / "textbooks.parquet"

    @computed_field
    @cached_property
    def IMAGES_DF_PATH(self) -> Path:
        return self.DFS_PATH / "images.parquet"

    @computed_field
    @cached_property
    def TEXT_BLOCKS_DF_PATH(self) -> Path:
        return self.DFS_PATH / "text_blocks.parquet"

    @computed_field
    @cached_property
    def CHUNKS_DF_PATH(self) -> Path:
        return self.DFS_PATH / "chunks.parquet"


settings = Settings()