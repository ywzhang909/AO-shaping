from __future__ import annotations

from pathlib import Path

import pytest

from ao_shaping.utils.io import paths

ENV_NAMES = (
    paths.ENV_DATA_DIR,
    paths.ENV_WEIGHTS_DIR,
    paths.ENV_LEGACY_WEIGHTS_DIR,
    paths.ENV_CONFIG_FILE,
) + tuple(env for env, _ in paths.DEVICE_CONFIG_KINDS.values())


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


# --- 设备配置目录 ----------------------------------------------------------


def test_device_config_dir_defaults_to_the_anchor_data_dir() -> None:
    """没配置时, 设备配置目录落在 ``<锚定目录>/data/<kind>_configs``。

    源码检出下锚定目录是仓库根, 所以这与改造前 ``PROJECT_ROOT / "data" / ...``
    逐位一致 —— 行为不变是这次改造的硬要求。
    """
    anchor = paths.anchor_dir()
    for kind, (_, default_name) in paths.DEVICE_CONFIG_KINDS.items():
        assert paths.device_config_dir(kind) == anchor / "data" / default_name


def test_device_config_dir_env_var_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_name, _ = paths.DEVICE_CONFIG_KINDS["ccd"]
    monkeypatch.setenv(env_name, str(tmp_path / "my_ccd"))

    assert paths.device_config_dir("ccd") == tmp_path / "my_ccd"


def test_device_config_dir_config_file_wins_over_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _write_config(tmp_path, '[paths]\nccd_config_dir = "cfg_ccd"\n')
    monkeypatch.setenv(paths.ENV_CONFIG_FILE, str(cfg))

    assert paths.device_config_dir("ccd") == (cfg.parent / "cfg_ccd").resolve()


def test_device_config_dir_env_var_beats_config_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _write_config(tmp_path, '[paths]\nccd_config_dir = "cfg_ccd"\n')
    monkeypatch.setenv(paths.ENV_CONFIG_FILE, str(cfg))
    monkeypatch.setenv("CCD_CONFIG_DIR", str(tmp_path / "env_ccd"))

    assert paths.device_config_dir("ccd") == tmp_path / "env_ccd"


def test_explicit_data_dir_pulls_the_device_config_dirs_along(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """显式配置了数据目录时, 设备配置目录默认跟着它走。

    "把数据放到另一块盘" 是一个整体动作; 拆成两半会让用户以为配置没生效。
    """
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "shared"))

    assert paths.device_config_dir("ccd") == tmp_path / "shared" / "ccd_configs"
    assert paths.weights_dir() == tmp_path / "shared" / "slm_models"


def test_explicit_device_config_dir_wins_over_the_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """显式写了某个设备配置目录时, 它不受数据目录牵连。"""
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "shared"))
    monkeypatch.setenv("CCD_CONFIG_DIR", str(tmp_path / "elsewhere"))

    assert paths.device_config_dir("ccd") == tmp_path / "elsewhere"
    # 其它设备仍然跟着数据目录
    assert paths.device_config_dir("slm") == tmp_path / "shared" / "slm_configs"


def test_unknown_device_config_kind_raises() -> None:
    with pytest.raises(ValueError, match="未知设备配置目录种类"):
        paths.device_config_dir("nope")


def test_config_file_is_re_read_when_it_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """配置文件改了立即生效 (按 mtime 失效缓存)。"""
    cfg = _write_config(tmp_path, '[paths]\ndata_dir = "first"\n')
    monkeypatch.setenv(paths.ENV_CONFIG_FILE, str(cfg))
    assert paths.data_dir() == (cfg.parent / "first").resolve()

    cfg.write_text('[paths]\ndata_dir = "second"\n', encoding="utf-8")
    # mtime 精度: 显式推进, 免得同一秒内写入被缓存吞掉。
    import os

    stat = cfg.stat()
    os.utime(cfg, (stat.st_atime, stat.st_mtime + 2))

    assert paths.data_dir() == (cfg.parent / "second").resolve()
