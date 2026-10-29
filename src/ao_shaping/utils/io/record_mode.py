from __future__ import annotations

from enum import Enum


class RecordMode(str, Enum):
    MAX = 'max'
    MIN = 'min'

    @classmethod
    def coerce(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v == 'max':
                return cls.MAX
            if v == 'min':
                return cls.MIN
        return cls.MIN

    @property
    def maximizing(self):
        return self is self.MAX

    @property
    def sign(self):
        return -1.0 if self is self.MAX else 1.0

    def __str__(self):
        return self.value
