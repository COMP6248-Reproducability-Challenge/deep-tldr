import logging
import sys
import os


def create_logger():
    logger = logging.getLogger('main_logger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(processName)s] %(levelname)s %(filename)s %(message)s')

    # Console, currently set to debug
    console_logger = logging.StreamHandler(sys.stdout)
    console_logger.setLevel(logging.DEBUG)
    console_logger.setFormatter(formatter)
    logger.addHandler(console_logger)
    return logger


logger = create_logger()