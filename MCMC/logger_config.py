import logging
from config import args


def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    '''
    file_handler = logging.FileHandler(path, mode='w')
    file_handler.setFormatter(log_format)
    logger.handlers = [file_handler]
    '''
    return logger


logger = _setup_logger()
#logger = _setup_logger(f'./log/{args.task}/{args.is_train}_{args.walks_num}walk_{args.N_negs}negative_degree_weighted_tail.log')