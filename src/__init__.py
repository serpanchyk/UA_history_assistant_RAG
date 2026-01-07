from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_PATH / "data"

DFS_PATH = DATA_PATH / "dfs"
TEXTBOOKS_DF_PATH = DFS_PATH / "textbooks.csv"
IMAGES_DF_PATH = DFS_PATH / "images.pkl"
TEXT_BLOKS_DF_PATH = DFS_PATH / "text_blocks.pkl"

TEXTBOOKS_DIR_PATH = DATA_PATH / "pdfs/"

IMAGES_DIR_PATH = DATA_PATH / "images"
REJECTED_IMAGES_DIR_PATH = IMAGES_DIR_PATH / "rejected_images"

IMAGES_DIR_PATH.mkdir(parents=True, exist_ok=True)
REJECTED_IMAGES_DIR_PATH.mkdir(parents=True, exist_ok=True)