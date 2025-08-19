import os
import sys
import logging
from datetime import datetime


class LoggerHandler:
    def __init__(self) -> None:
        root_dir = os.getcwd()
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = os.path.join(root_dir, f"logs/output_{current_time}.log")

    def get_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        log_file = self.log_path
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        terminal_handler = logging.StreamHandler(sys.stdout)
        terminal_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        terminal_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(terminal_handler)

        return logger
