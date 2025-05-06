import os
from config import args
import logging


def _setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")

    stdout_handler = logging.StreamHandler()
    # stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_fpath = os.path.join(args.log_dir, 'logs.log')
    dir_path = os.path.dirname(file_fpath)
    # 确保目录存在
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    file_handler = logging.FileHandler(file_fpath)
    # file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger


logger = _setup_logger()
