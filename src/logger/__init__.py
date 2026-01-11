import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from pathlib import Path

now = datetime.now()

this_file_path = Path(__file__)

# __init__ -> logger -> src -> project
repo_path = this_file_path.parent.parent.parent

# Folder for logs: log/YYYY/MM
log_path = repo_path / 'log' / str(now.year) / f"{now.month:02d}"
log_path.mkdir(parents=True, exist_ok=True)


formatter = logging.Formatter(
    fmt='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

log_file = log_path /  f"logger_{now.day:02d}.log"
file_handler = TimedRotatingFileHandler(
    log_file,
    when='d',
    interval=1,
    encoding='utf-8'
)
file_handler.setFormatter(formatter)

logger = logging.getLogger("rag_pipeline")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)