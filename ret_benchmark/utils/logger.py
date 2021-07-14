import os
import sys
import logging
import os.path as osp
_streams = {"stdout": sys.stdout}


# def setup_logger(name: str, level: int, stream: str = "stdout") -> logging.Logger:
#     global _streams
#     if stream not in _streams:
#         log_folder = os.path.dirname(stream)
#         os.makedirs(log_folder, exist_ok=True)
#         _streams[stream] = open(stream, "w")
#     logger = logging.getLogger(name)
#     logger.propagate = False
#     logger.setLevel(level)

#     sh = logging.StreamHandler(stream=_streams[stream])
#     sh.setLevel(level)

#     fh = logging.FileHandler()
#     formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
#     sh.setFormatter(formatter)
#     logger.addHandler(sh)
#     return logger


def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if if_train:
            fh = logging.FileHandler(os.path.join(save_dir, name+"train_log.txt"), mode='w')
        else:
            fh = logging.FileHandler(os.path.join(save_dir, name+"test_log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger