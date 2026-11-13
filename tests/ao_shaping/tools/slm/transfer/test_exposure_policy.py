"""开场曝光种子策略 (:mod:`ao_shaping.tools.slm.transfer.exposure_policy`) 的行为锁。

只测策略本身 (词汇 → 种子 / 生效值 / 记录文件), 不碰设备: 相机一律用桩。
"""

from __future__ import annotations

import argparse
import json
import os

import pytest
from click import BadParameter

from ao_shaping.tools.slm.transfer import exposure_policy as policy
from ao_shaping.utils.cli.exposure import AUTO_EXPOSURE, DEFAULT_EXPOSURE


class _StubCamera:
    """只回答「我这台相机现在实际是多少 ms」的最小桩。"""

    def __init__(self, effective_ms: float | None = 1.25) -> None:
        self._effective = effective_ms
        self.calls = 0

    def get_exposure_time_ms(self) -> float:
        self.calls += 1
        if self._effective is None:
            raise RuntimeError("SDK 没回话")
        return self._effective


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """把数据目录指到 tmp, 记录文件于是与仓库 ``data/`` 无关。"""
    monkeypatch.setenv("AO_SHAPING_DATA_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# 三态: 数字 / auto / default
# ---------------------------------------------------------------------------


def test_explicit_number_is_the_seed(data_root):
    assert policy.resolve_seed_exposure("0.5") == 0.5
    assert policy.resolve_seed_exposure("3.25") == 3.25
    # 数字形式的原始值 (GUI / 会话对象传进来的浮点) 走同一条路
    assert policy.resolve_seed_exposure(0.5) == 0.5


def test_auto_without_record_falls_back_to_the_old_hardcoded_value(data_root):
    assert policy.resolve_seed_exposure(AUTO_EXPOSURE, camera="daheng", cam_id=0) == (
        policy.FALLBACK_EXPOSURE_MS
    )
    # None 与 "auto" 同义 (默认策略)
    assert policy.resolve_seed_exposure(None, camera="daheng") == (
        policy.FALLBACK_EXPOSURE_MS
    )


def test_auto_reads_the_remembered_effective_exposure(data_root):
    policy.remember_exposure("daheng", 0, 0.4)
    assert policy.resolve_seed_exposure(AUTO_EXPOSURE, camera="daheng") == 0.4


def test_auto_is_per_camera_and_per_cam_id(data_root):
    policy.remember_exposure("daheng", 0, 0.4)
    policy.remember_exposure("daheng", 1, 2.0)
    policy.remember_exposure("miicam", 0, 8.0)
    assert policy.resolve_seed_exposure(AUTO_EXPOSURE, camera="daheng", cam_id=0) == 0.4
    assert policy.resolve_seed_exposure(AUTO_EXPOSURE, camera="daheng", cam_id=1) == 2.0
    assert policy.resolve_seed_exposure(AUTO_EXPOSURE, camera="miicam", cam_id=0) == 8.0


def test_state_key_is_stable_across_processes(data_root):
    """``hash()`` 每次解释器加盐: 用它当文件名等于「下次开机找不到记录」。"""
    import zlib

    digest = format(zlib.crc32(b"daheng-0"), "08x")
    assert policy.state_path("daheng", 0).name == f"daheng-0_{digest}.json"
    assert policy.state_path("daheng", 0).name == policy.state_path("DAHENG", 0).name


def test_default_means_do_not_touch_the_camera(data_root):
    """``default`` 必须是 None (= 连关键字都不给), 不是「回落 0.02」。"""
    assert policy.resolve_seed_exposure(DEFAULT_EXPOSURE, camera="daheng") is None
    assert policy.resolve_seed_exposure("default") is None
    assert not (data_root / ".cache" / "slm_transfer_exposure").exists()


def test_auto_seeds_from_record_after_a_ladder_even_when_policy_was_default(data_root):
    """default 会话收住阶梯后照样更新状态, 下一次 auto 因此拿到真值。"""
    camera = _StubCamera(effective_ms=2.5)
    policy.record_effective_exposure(camera, 0, 0.02, backend="daheng")
    assert policy.resolve_seed_exposure(AUTO_EXPOSURE, camera="daheng") == 2.5


def test_seed_is_lifted_to_the_device_floor(data_root):
    assert policy.resolve_seed_exposure("0.001", floor_ms=0.05) == 0.05
    policy.remember_exposure("daheng", 0, 0.001)
    assert policy.resolve_seed_exposure(
        AUTO_EXPOSURE, camera="daheng", floor_ms=0.05
    ) == 0.05


@pytest.mark.parametrize("raw", ["0", "-3", "nonsense", "nan", "inf", "-0.0"])
def test_bad_tokens_are_rejected_before_any_device_work(data_root, raw):
    with pytest.raises(BadParameter):
        policy.resolve_seed_exposure(raw, camera="daheng")
    with pytest.raises(BadParameter):
        policy.resolve_cli_exposure(raw, param_hint="--exposure-ms")


# ---------------------------------------------------------------------------
# 生效曝光 (落盘用的那一个)
# ---------------------------------------------------------------------------


def test_effective_exposure_prefers_the_device_read_back(data_root):
    camera = _StubCamera(effective_ms=1.25)
    assert policy.resolve_effective_exposure(camera, 0.02) == 1.25
    assert camera.calls == 1


def test_effective_exposure_degrades_to_the_request(data_root):
    camera = _StubCamera(effective_ms=None)  # 回读炸了
    assert policy.resolve_effective_exposure(camera, 0.5) == 0.5
    assert policy.resolve_effective_exposure(None, 0.5) == 0.5
    assert policy.resolve_effective_exposure(None, None) is None


def test_record_effective_exposure_stores_the_read_back(data_root):
    camera = _StubCamera(effective_ms=3.5)
    assert policy.record_effective_exposure(camera, 0, 0.02, backend="daheng") == 3.5
    assert policy.read_effective_exposure("daheng", 0) == 3.5
    # 阶梯结束后相机停在 0.7: 记的是 0.7, 不是请求值 1.0
    camera._effective = 0.7
    assert policy.record_effective_exposure(camera, 0, 1.0, backend="daheng") == 0.7
    assert policy.read_effective_exposure("daheng", 0) == 0.7


def test_record_effective_exposure_with_a_backend_name(data_root):
    assert policy.record_effective_exposure("miicam", 2, 1.5) == 1.5
    assert policy.read_effective_exposure("miicam", 2) == 1.5
    assert policy.read_effective_exposure("miicam", 3) is None


def test_record_effective_exposure_without_a_usable_value_writes_nothing(data_root):
    """没有可用生效曝光 (回读失败且无请求值) 时不写记录, 免得污染种子。"""
    assert policy.record_effective_exposure(None, 0, None, backend="daheng") is None
    assert policy.read_effective_exposure("daheng", 0) is None


# ---------------------------------------------------------------------------
# 记录文件本身
# ---------------------------------------------------------------------------


def test_remember_read_clear_round_trip(data_root):
    assert policy.read_effective_exposure("daheng", 0) is None
    assert policy.remember_exposure("daheng", 0, 0.4) == 0.4
    assert policy.read_effective_exposure("daheng", 0) == 0.4
    payload = json.loads(policy.state_path("daheng", 0).read_text(encoding="utf-8"))
    assert payload["exposure_ms"] == 0.4
    assert payload["camera"] == "daheng"
    assert payload["cam_id"] == 0
    assert "updated_at" in payload
    policy.clear_exposure_state("daheng", 0)
    assert policy.read_effective_exposure("daheng", 0) is None


@pytest.mark.parametrize("value", [None, 0, -1, "nonsense", float("nan")])
def test_unusable_values_are_never_remembered(data_root, value):
    assert policy.remember_exposure("daheng", 0, value) is None
    assert not policy.state_path("daheng", 0).exists()


def test_corrupt_state_reads_as_no_record(data_root):
    path = policy.state_path("daheng", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    assert policy.read_effective_exposure("daheng", 0) is None
    assert policy.resolve_seed_exposure(AUTO_EXPOSURE, camera="daheng", cam_id=0) == (
        policy.FALLBACK_EXPOSURE_MS
    )
    # 坏文件会被下一次记录原子覆盖掉
    assert policy.remember_exposure("daheng", 0, 0.6) == 0.6
    assert policy.read_effective_exposure("daheng", 0) == 0.6


def test_non_dict_state_reads_as_no_record(data_root):
    path = policy.state_path("daheng", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([0.4]), encoding="utf-8")
    assert policy.read_effective_exposure("daheng", 0) is None


def test_temp_file_is_left_behind_never(data_root):
    """原子改名的临时文件既不该留下, 也不该被当成记录读到。"""
    policy.remember_exposure("daheng", 0, 0.4)
    directory = policy.state_path("daheng", 0).parent
    assert not list(directory.glob("*.tmp"))
    assert len(list(directory.glob("*.json"))) == 1


@pytest.mark.skipif(os.name == "nt", reason="Windows ACLs 不让 mkdir 稳定失败")
def test_unwritable_data_root_degrades_instead_of_breaking_the_session(
    data_root, monkeypatch
):
    """数据目录只读: 记不下只是少一条种子, 采集照跑。"""

    class _ReadOnly:
        # 真实 Path 必有 name；remember_exposure 在 mkdir 之前用它拼临时文件名，
        # 桩缺这个名字会在进入 try 之前抛 AttributeError，测的就不再是"只读目录"。
        name = "slm_transfer_exposure.json"

        @property
        def parent(self):
            return self

        def mkdir(self, *args, **kwargs):
            raise PermissionError("read-only data dir")

        def with_name(self, name):
            return self

        def write_bytes(self, payload):
            raise PermissionError("read-only data dir")

        def replace(self, target):
            raise PermissionError("read-only data dir")

        def unlink(self, *args, **kwargs):
            raise PermissionError("read-only data dir")

        def read_text(self, encoding=None):
            raise PermissionError("read-only data dir")

    monkeypatch.setattr(policy, "state_path", lambda *a, **k: _ReadOnly())
    assert policy.remember_exposure("daheng", 0, 0.4) is None
    assert policy.read_effective_exposure("daheng", 0) is None
    assert policy.resolve_seed_exposure(AUTO_EXPOSURE, camera="daheng", cam_id=0) == (
        policy.FALLBACK_EXPOSURE_MS
    )
    policy.clear_exposure_state("daheng", 0)  # 不抛


def test_state_file_name_is_filesystem_safe(data_root):
    path = policy.state_path("DA HENG:2/0", 3)
    assert path.parent == data_root / ".cache" / "slm_transfer_exposure"
    assert "/" not in path.name and "\\" not in path.name and ":" not in path.name
    # 不同 (后端, 序号) 不撞同一个文件
    assert policy.state_path("daheng", 0) != policy.state_path("daheng", 1)
    assert policy.state_path("daheng", 0) != policy.state_path("miicam", 0)


# ---------------------------------------------------------------------------
# CLI 词汇层 (不新造拼写)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "auto"),
        ("", "auto"),
        ("Auto", "auto"),
        (" auto ", "auto"),
        ("default", "default"),
        ("DEFAULT", "default"),
        ("0.5", "0.5"),
        (0.5, "0.5"),
        (" 2 ", "2.0"),
        (AUTO_EXPOSURE, "auto"),  # 已解析的值可往返
        (DEFAULT_EXPOSURE, "default"),
    ],
)
def test_cli_tokens_are_normalized_without_new_spellings(data_root, raw, expected):
    assert policy.resolve_cli_exposure(raw) == expected


def test_cli_rejects_values_that_are_not_a_seed(data_root):
    for raw in ("0", "-1", "0.0", "abc", True):
        with pytest.raises(BadParameter):
            policy.resolve_cli_exposure(raw, param_hint="--exposure-ms")


def test_argparse_type_wraps_rejection_as_usage_error(data_root):
    parse = policy.argparse_exposure_type("--exposure-ms")
    assert parse("auto") == "auto"
    assert parse("default") == "default"
    assert parse("1.5") == "1.5"
    with pytest.raises(argparse.ArgumentTypeError):
        parse("0")


@pytest.mark.parametrize(
    ("raw", "source"),
    [
        (None, "auto"),
        ("auto", "auto"),
        ("default", "default"),
        ("0.5", "explicit"),
        ("nonsense", "auto"),  # 认不出 → 报默认来源, 绝不抛
    ],
)
def test_policy_source_labels(data_root, raw, source):
    assert policy.policy_source(raw) == source


def test_fallback_is_the_value_that_was_hardcoded(data_root):
    assert policy.FALLBACK_EXPOSURE_MS == 0.02
