import re
from datetime import datetime
from pathlib import Path

import click
import pytest

from ao_shaping.utils.cli_helpers import (
    parse_tuple,
    setup_coredumpy,
    get_date_dir_name,
    get_timestamp_str,
    create_save_dir,
)


class TestParseTuple:
    def test_none_returns_none(self):
        assert parse_tuple(None, None, None) is None

    def test_simple_tuple(self):
        assert parse_tuple(None, None, "100,200") == (100, 200)

    def test_parenthesized_tuple(self):
        assert parse_tuple(None, None, "(100,200)") == (100, 200)

    def test_parenthesized_with_spaces(self):
        assert parse_tuple(None, None, "( 100, 200 )") == (100, 200)

    def test_string_mass(self):
        assert parse_tuple(None, None, "mass") == "mass"

    def test_string_max(self):
        assert parse_tuple(None, None, "max") == "max"

    def test_string_shape(self):
        assert parse_tuple(None, None, "shape") == "shape"

    def test_string_case_insensitive(self):
        assert parse_tuple(None, None, "MASS") == "mass"

    def test_invalid_single_value(self):
        with pytest.raises(click.BadParameter):
            parse_tuple(None, None, "100")

    def test_invalid_non_numeric(self):
        with pytest.raises(click.BadParameter):
            parse_tuple(None, None, "abc,def")

    def test_invalid_three_values(self):
        with pytest.raises(click.BadParameter):
            parse_tuple(None, None, "1,2,3")


class TestGetDateDirName:
    def test_format(self):
        result = get_date_dir_name()
        assert re.match(r"\d{8}", result)

    def test_matches_today(self):
        result = get_date_dir_name()
        expected = datetime.now().strftime("%Y%m%d")
        assert result == expected


class TestGetTimestampStr:
    def test_format(self):
        result = get_timestamp_str()
        assert re.match(r"\d{8}_\d{6}", result)


class TestSetupCoredumpy:
    def test_returns_bool(self):
        result = setup_coredumpy()
        assert isinstance(result, bool)


class TestCreateSaveDir:
    def test_creates_directory(self, tmp_path):
        result = create_save_dir(tmp_path, "test_subdir")
        assert result.exists()
        assert result.is_dir()

    def test_includes_date_subdir(self, tmp_path):
        result = create_save_dir(tmp_path, "test_subdir")
        date_str = get_date_dir_name()
        assert date_str in str(result)

    def test_with_path_object(self, tmp_path):
        result = create_save_dir(Path(tmp_path), "wf")
        assert result.exists()

    def test_idempotent(self, tmp_path):
        create_save_dir(tmp_path, "test_subdir")
        result = create_save_dir(tmp_path, "test_subdir")
        assert result.exists()
