from .spots_calc import *
from .file import *
from .display import ImageVoltagesDisplay
from .timestamp import TimestampParser, parse_timestamp, sort_by_timestamp

from loguru import logger

error_handler = logger.add(
    "logs/error.log",
    rotation="500 MB",
    encoding="utf-8",
    level="ERROR",
    backtrace=True,
    diagnose=True,
)
