import logging
import os
from pathlib import Path
from app.configs import settings


logs_dir = Path(settings.LOGS_DIR)
logs_dir.mkdir(exist_ok=True)

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(settings.LOGS_DIR + "/app.log")
console_handler = logging.StreamHandler()

log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

