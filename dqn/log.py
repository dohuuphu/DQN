import logging

from dqn.variables import *
from logging.handlers import RotatingFileHandler

# Init logger
class Logger():
    def __init__(self) -> None:
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%m/%d/%Y %I:%M:%S %p')
        self.setup_logger(SYSTEM_LOG, SYSTEM_PATH)
        self.setup_logger(RECOMMEND_LOG, RECOMMEND_PATH)
        self.setup_logger(CHECKDONE_LOG, CHECKDONE_PATH)

    def setup_logger(self, name, log_file, level=logging.DEBUG):
        """To setup as many loggers as you want"""

        my_handler = RotatingFileHandler(log_file, mode='a', maxBytes=10*1024*1024, 
                                 backupCount=5, encoding=None, delay=0)
        # handler = logging.FileHandler(log_file)        
        my_handler.setFormatter(self.formatter)
        

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.addHandler(my_handler)

        return logger_