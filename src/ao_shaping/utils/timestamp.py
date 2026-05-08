"""Timestamp parsing utilities for image files and filenames.

This module provides comprehensive timestamp parsing from various filename formats
and datetime representations. It supports multiple common timestamp patterns used
in scientific imaging and data acquisition systems.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


class TimestampParser:
    """Parse timestamps from various filename formats.

    Supports multiple common timestamp patterns:
    - Unix epoch milliseconds: 1700000000000.png
    - Unix epoch seconds: 1700000000.png
    - Unix epoch with microseconds: 1700000000000000.png
    - DateTime strings: 20240101_120000.png, 2024-01-01_12-00-00.png
    - ISO format: 2024-01-01T12-00-00.png
    - Custom formats with strptime

    Example:
        >>> parser = TimestampParser()
        >>> timestamp = parser.parse("1700000000000.png")
        >>> print(timestamp)
        2023-11-14 22:13:20

        >>> ts, filename = parser.parse_with_timestamp("20240101_120000.png")
        >>> print(ts)
        2024-01-01 12:00:00
    """

    # Regex patterns for common timestamp formats
    PATTERNS: list[tuple[str, re.Pattern, str]] = [
        (
            "unix_ms",
            re.compile(r"^(\d{13})(?:_[a-zA-Z0-9]+)?$"),
            "unix_ms",
        ),
        (
            "unix_s",
            re.compile(r"^(\d{10})(?:_[a-zA-Z0-9]+)?$"),
            "unix_s",
        ),
        (
            "unix_us",
            re.compile(r"^(\d{16})(?:_[a-zA-Z0-9]+)?$"),
            "unix_us",
        ),
        (
            "datetime_compact",
            re.compile(r"^(\d{8})_(\d{6})(?:_[a-zA-Z0-9]+)?$"),
            "%Y%m%d_%H%M%S",
        ),
        (
            "datetime_dashes",
            re.compile(
                r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})(?:_[a-zA-Z0-9]+)?$"
            ),
            "%Y-%m-%d_%H-%M-%S",
        ),
        (
            "date_only",
            re.compile(r"^(\d{8})(?:_[a-zA-Z0-9]+)?$"),
            "%Y%m%d",
        ),
        (
            "iso_datetime",
            re.compile(
                r"^(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})(?:_[a-zA-Z0-9]+)?$"
            ),
            "%Y-%m-%dT%H-%M-%S",
        ),
        (
            "datetime_dots",
            re.compile(
                r"^(\d{4})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})(?:_[a-zA-Z0-9]+)?$"
            ),
            "%Y.%m.%d.%H.%M.%S",
        ),
        (
            "time_only",
            re.compile(r"^(\d{6})(?:_[a-zA-Z0-9]+)?$"),
            "%H%M%S",
        ),
    ]

    # Custom strptime formats to try
    CUSTOM_FORMATS: list[str] = [
        "%Y%m%d_%H%M%S.%f",
        "%Y-%m-%d_%H-%M-%S.%f",
        "%Y%m%d_%H%M%S",
        "%Y-%m-%d_%H:%M:%S",
        "%Y%m%d%H%M%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y_%m_%d_%H_%M_%S",
    ]

    def __init__(self, default_date: datetime | None = None):
        """Initialize parser.

        Args:
            default_date: Default date to use when only time is present.
                If None, uses today's date.
        """
        self.default_date = default_date

    def parse(self, filename: str) -> datetime | None:
        """Parse timestamp from filename.

        Args:
            filename: The filename to parse (can include path).

        Returns:
            datetime object if parsing successful, None otherwise.
        """
        # Extract basename if path provided
        name = Path(filename).stem

        # Try each pattern
        for pattern_name, pattern, fmt in self.PATTERNS:
            match = pattern.match(name)
            if match:
                if fmt == "unix_ms":
                    ts = int(match.group(1))
                    return datetime.fromtimestamp(ts / 1000)
                elif fmt == "unix_s":
                    ts = int(match.group(1))
                    return datetime.fromtimestamp(ts)
                elif fmt == "unix_us":
                    ts = int(match.group(1))
                    return datetime.fromtimestamp(ts / 1_000_000)
                else:
                    last_group_end = max(
                        match.end(i)
                        for i in range(1, len(match.groups()) + 1)
                        if match.group(i) is not None
                    )
                    ts_portion = name[:last_group_end]
                    try:
                        return datetime.strptime(ts_portion, fmt)
                    except ValueError:
                        continue

        # Try custom formats
        for fmt in self.CUSTOM_FORMATS:
            try:
                return datetime.strptime(name, fmt)
            except ValueError:
                continue

        return None

    def parse_with_timestamp(self, filename: str) -> tuple[datetime, str]:
        """Parse timestamp and return with original filename.

        Args:
            filename: The filename to parse.

        Returns:
            Tuple of (datetime, original_filename_stem).
            If parsing fails, returns (None, original_stem).
        """
        stem = Path(filename).stem
        ts = self.parse(filename)
        if ts is None:
            # Return original stem if parse fails
            return datetime.now(), stem
        return ts, stem

    def sort_files(self, files: list[str | Path], reverse: bool = False) -> list[Path]:
        """Sort files by timestamp parsed from filenames.

        Args:
            files: List of file paths or filenames to sort.
            reverse: If True, sort in descending order.

        Returns:
            Sorted list of Path objects.
        """

        def get_timestamp(path: Path) -> float:
            ts = self.parse(path.name)
            if ts is not None:
                return ts.timestamp()
            # If no timestamp found, use mtime as fallback
            try:
                return path.stat().st_mtime
            except OSError:
                return 0

        return sorted(
            [Path(f) for f in files],
            key=get_timestamp,
            reverse=reverse,
        )

    @staticmethod
    def format_timestamp(ts: datetime, fmt: str = "%Y%m%d_%H%M%S") -> str:
        """Format datetime to string.

        Args:
            ts: datetime object to format.
            fmt: Output format (strptime-style).

        Returns:
            Formatted string.
        """
        return ts.strftime(fmt)

    @staticmethod
    def generate_filename(
        ts: datetime,
        prefix: str = "",
        suffix: str = "",
        extension: str = "png",
    ) -> str:
        """Generate filename from datetime.

        Args:
            ts: datetime object.
            prefix: Prefix to add (e.g., "image_").
            suffix: Suffix to add before extension (e.g., "_001").
            extension: File extension (without dot).

        Returns:
            Generated filename.
        """
        ts_str = ts.strftime("%Y%m%d_%H%M%S")
        if prefix or suffix:
            return f"{prefix}{ts_str}{suffix}.{extension}"
        return f"{ts_str}.{extension}"


def parse_timestamp(filename: str) -> datetime | None:
    """Convenience function to parse timestamp from filename.

    Args:
        filename: The filename to parse.

    Returns:
        datetime object if parsing successful, None otherwise.
    """
    return TimestampParser().parse(filename)


def sort_by_timestamp(files: list[str | Path]) -> list[Path]:
    """Convenience function to sort files by timestamp.

    Args:
        files: List of file paths or filenames.

    Returns:
        Sorted list of Path objects.
    """
    return TimestampParser().sort_files(files)
