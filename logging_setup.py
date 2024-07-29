import logging
import os

def setup_logging(log_filename):
    if os.path.exists(log_filename):
        os.remove(log_filename)
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)
