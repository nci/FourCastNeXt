import sys
import logging

def get_logger(name):
  logger_name = f'_logger_{name}'
  if logger_name in sys.modules:
    return sys.modules[logger_name]

  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)
  logger.propagate = False

  # create formatter
  fmt = '[%(asctime)s %(filename)s %(lineno)d %(levelname)s]: %(message)s'

  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.setLevel(logging.DEBUG)
  console_handler.setFormatter(
      logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
  logger.addHandler(console_handler)

  sys.modules[logger_name] = logger
  return logger

