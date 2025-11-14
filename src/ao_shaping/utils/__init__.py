from .spots_calc import *
from .file import *
from .display import ImageVoltagesDisplay

from loguru import logger
error_handler = logger.add("logs/error.log", rotation="500 MB", encoding="utf-8", level="ERROR", backtrace=True, diagnose=True)