from __future__ import annotations

from pathlib import Path

import pytest

from ao_shaping.utils.io import paths

ENV_NAMES = (
    paths.ENV_DATA_DIR,
    paths.ENV_WEIGHTS_DIR,
    paths.ENV_LEGACY_WEIGHTS_DIR,
    paths.ENV_CONFIG_FILE,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    # 把配置文件默认位置指向一个不存在的文件, 避免读到仓库里的真实配置。
    monkeypatch.setenv(paths.ENV_CONFIG_FILE, str(tmp_path / "none.toml"))


def _write_config(tmp_path: Path, body: str) -> Path:
    cfg = tmp_path / "cfg" / "ao_shaping.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(body, encoding="utf-8")
    return cfg


def test_defaults_when_nothing_configured() -> None:
    assert paths.data_dir() == Path("data")
    assert paths.weights_dir().name == "slm_models"
    assert paths.weights_dir().parent.name == "data"


def test_env_overrides_config_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, '[paths]\ndata_dir = "from_cfg"\nweights_dir = "w_cfg"\n')
    monkeypatch.setenv(paths.ENV_CONFIG_FILE, str(cfg))
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "env_data"))
    monkeypatch.setenv(paths.ENV_WEIGHTS_DIR, str(tmp_path / "env_w"))

    assert paths.data_dir() == tmp_path / "env_data"
    assert paths.weights_dir() == tmp_path / "env_w"


def test_legacy_slm_models_env_beats_new_weights_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(paths.ENV_WEIGHTS_DIR, str(tmp_path / "new"))
    monkeypatch.setenv(paths.ENV_LEGACY_WEIGHTS_DIR, str(tmp_path / "legacy"))

    assert paths.weights_dir() == tmp_path / "legacy"


def test_config_file_relative_paths_resolve_against_config_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _write_config(tmp_path, '[paths]\ndata_dir = "../shared_data"\n')
    monkeypatch.setenv(paths.ENV_CONFIG_FILE, str(cfg))

    assert paths.data_dir() == (cfg.parent / ".." / "shared_data").resolve()
    # 权重目录未配置 → 落在数据目录下
    assert paths.weights_dir() == (cfg.parent / ".." / "shared_data").resolve() / "slm_models"


def test_config_file_absolute_weights_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    w = tmp_path / "big_disk" / "models"
    cfg = _write_config(tmp_path, f'[paths]\nweights_dir = "{w.as_posix()}"\n')
    monkeypatch.setenv(paths.ENV_CONFIG_FILE, str(cfg))

    assert paths.weights_dir() == w


def test_malformed_config_falls_back_to_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, "[paths\nbroken = ")
    monkeypatch.setenv(paths.ENV_CONFIG_FILE, str(cfg))

    assert paths.data_dir() == Path("data")


def test_blank_env_value_is_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, '[paths]\ndata_dir = "cfg_data"\n')
    monkeypatch.setenv(paths.ENV_CONFIG_FILE, str(cfg))
    monkeypatch.setenv(paths.ENV_DATA_DIR, "   ")

    assert paths.data_dir() == (cfg.parent / "cfg_data").resolve()
