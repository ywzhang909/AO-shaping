"""Tests for ao_shaping.utils.timestamp — TimestampParser and helpers."""

from datetime import datetime

import pytest

from ao_shaping.utils.timestamp import (
    TimestampParser,
    parse_timestamp,
    sort_by_timestamp,
)


class TestTimestampParser:
    def test_parse_unix_ms(self):
        ts = TimestampParser().parse("1700000000000.png")
        assert ts is not None
        assert ts.year == 2023

    def test_parse_unix_s(self):
        ts = TimestampParser().parse("1700000000.png")
        assert ts is not None
        assert ts.year == 2023

    def test_parse_unix_us(self):
        ts = TimestampParser().parse("1700000000000000.png")
        assert ts is not None
        assert ts.year == 2023

    def test_parse_datetime_compact(self):
        ts = TimestampParser().parse("20240101_120000.png")
        assert ts is not None
        assert ts.year == 2024
        assert ts.month == 1
        assert ts.hour == 12

    def test_parse_datetime_compact_with_path(self):
        ts = TimestampParser().parse("/data/images/20240101_120000.png")
        assert ts is not None
        assert ts.year == 2024

    def test_parse_datetime_compact_with_suffix(self):
        ts = TimestampParser().parse("20240101_120000_ref.png")
        assert ts is not None
        assert ts.year == 2024

    def test_parse_date_only(self):
        ts = TimestampParser().parse("20240101.png")
        assert ts is not None
        assert ts.year == 2024

    def test_parse_datetime_dashes(self):
        ts = TimestampParser().parse("2024-01-01_12-00-00.png")
        assert ts is not None
        assert ts.year == 2024
        assert ts.hour == 12

    def test_parse_iso_datetime(self):
        ts = TimestampParser().parse("2024-01-01T12-00-00.png")
        assert ts is not None
        assert ts.year == 2024

    def test_parse_unrecognized_returns_none(self):
        ts = TimestampParser().parse("random_file.txt")
        assert ts is None

    def test_parse_custom_format_hyphen_colon(self):
        ts = TimestampParser().parse("2024-06-15_10:30:00.png")
        assert ts is not None
        assert ts.year == 2024
        assert ts.month == 6

    def test_parse_custom_format_iso_colon(self):
        ts = TimestampParser().parse("2024-06-15T10:30:00.png")
        assert ts is not None
        assert ts.year == 2024


class TestParseWithTimestamp:
    def test_returns_tuple(self):
        ts, stem = TimestampParser().parse_with_timestamp("20240101_120000.png")
        assert isinstance(ts, datetime)
        assert stem == "20240101_120000"

    def test_unrecognized_returns_now(self):
        ts, stem = TimestampParser().parse_with_timestamp("unknown.txt")
        assert isinstance(ts, datetime)
        assert stem == "unknown"


class TestSortFiles:
    def test_sort_ascending(self):
        files = [
            "20240103_120000.png",
            "20240101_120000.png",
            "20240102_120000.png",
        ]
        sorted_files = TimestampParser().sort_files(files)
        names = [f.name for f in sorted_files]
        assert names[0] == "20240101_120000.png"
        assert names[2] == "20240103_120000.png"

    def test_sort_descending(self):
        files = [
            "20240101_120000.png",
            "20240103_120000.png",
            "20240102_120000.png",
        ]
        sorted_files = TimestampParser().sort_files(files, reverse=True)
        names = [f.name for f in sorted_files]
        assert names[0] == "20240103_120000.png"


class TestFormatTimestamp:
    def test_default_format(self):
        ts = datetime(2024, 1, 15, 10, 30, 0)
        result = TimestampParser.format_timestamp(ts)
        assert result == "20240115_103000"

    def test_custom_format(self):
        ts = datetime(2024, 6, 1, 12, 0, 0)
        result = TimestampParser.format_timestamp(ts, fmt="%Y-%m-%d")
        assert result == "2024-06-01"


class TestGenerateFilename:
    def test_basic(self):
        ts = datetime(2024, 1, 15, 10, 30, 0)
        result = TimestampParser.generate_filename(ts)
        assert result == "20240115_103000.png"

    def test_with_prefix(self):
        ts = datetime(2024, 1, 15, 10, 30, 0)
        result = TimestampParser.generate_filename(ts, prefix="img_")
        assert result == "img_20240115_103000.png"

    def test_with_suffix_and_extension(self):
        ts = datetime(2024, 1, 15, 10, 30, 0)
        result = TimestampParser.generate_filename(ts, suffix="_001", extension="tif")
        assert result == "20240115_103000_001.tif"


class TestConvenienceFunctions:
    def test_parse_timestamp(self):
        ts = parse_timestamp("20240101_120000.png")
        assert ts is not None
        assert ts.year == 2024

    def test_sort_by_timestamp(self):
        files = ["20240103_120000.png", "20240101_120000.png"]
        sorted_files = sort_by_timestamp(files)
        assert sorted_files[0].name == "20240101_120000.png"
