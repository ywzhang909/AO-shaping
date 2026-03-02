from .spots_calc import *
from .file import *

from loguru import logger
error_handler = logger.add("logs/error.log", rotation="500 MB", encoding="utf-8", level="ERROR", backtrace=True, diagnose=True)


class Register:
    def __init__(self) -> None:
        self.members = {}

    def register(self, name:str) -> None:
        def decorator(func):
            self.members[name] = func
            return func
        return decorator
    
    def __getitem__(self, name:str):
        return self.members[name]
    
    @property
    def all_funcs(self):
        return self.members.values()
    
    @property
    def all_names(self):
        return self.members.keys()