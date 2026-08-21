"""ioc.yaml 配置加载/校验单测。"""
from __future__ import annotations

import pytest

from ao_epics_common.ioc_config import (
    DeviceSpec,
    IocConfigError,
    IocSpec,
    load_ioc_config,
)

VALID_YAML = """
ioc:
  name: ioc-slm
  description: "test"
  prefix: "SLM-01:"
  ca_port: 5065
  devices:
    - name: slm
      type: santec_slm200
      params:
        slm_number: 1
        wavelength: 1064
        memory_slots: [3, 4, 5]
  pvs:
    - {name: Wavelength, type: int}
    - {name: PhasePattern, type: waveform}
"""


def _write(tmp_path, text: str):
    p = tmp_path / "ioc.yaml"
    p.write_text(text, encoding="utf-8")
    return p


class TestLoadIocConfig:
    def test_valid(self, tmp_path) -> None:
        spec = load_ioc_config(_write(tmp_path, VALID_YAML))
        assert isinstance(spec, IocSpec)
        assert spec.name == "ioc-slm"
        assert spec.prefix == "SLM-01:"
        assert spec.ca_port == 5065
        assert len(spec.devices) == 1
        dev = spec.devices[0]
        assert isinstance(dev, DeviceSpec)
        assert dev.name == "slm"
        assert dev.type == "santec_slm200"
        assert dev.params == {"slm_number": 1, "wavelength": 1064, "memory_slots": [3, 4, 5]}
        assert spec.pv_names == ["SLM-01:Wavelength", "SLM-01:PhasePattern"]

    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(IocConfigError, match="不存在"):
            load_ioc_config(tmp_path / "nope.yaml")

    def test_missing_required_field(self, tmp_path) -> None:
        text = VALID_YAML.replace("  ca_port: 5065\n", "")
        with pytest.raises(IocConfigError, match="缺少字段"):
            load_ioc_config(_write(tmp_path, text))

    def test_bad_yaml(self, tmp_path) -> None:
        with pytest.raises(IocConfigError, match="解析失败"):
            load_ioc_config(_write(tmp_path, "ioc: [unclosed"))

    def test_no_ioc_section(self, tmp_path) -> None:
        with pytest.raises(IocConfigError, match="必须含 ioc"):
            load_ioc_config(_write(tmp_path, "other: 1\n"))

    def test_ca_port_out_of_range(self, tmp_path) -> None:
        text = VALID_YAML.replace("ca_port: 5065", "ca_port: 99999")
        with pytest.raises(IocConfigError, match="ca_port 越界"):
            load_ioc_config(_write(tmp_path, text))

    def test_duplicate_pv_names(self, tmp_path) -> None:
        text = VALID_YAML.replace(
            "    - {name: PhasePattern, type: waveform}",
            "    - {name: Wavelength, type: waveform}",
        )
        with pytest.raises(IocConfigError, match="PV 名重复"):
            load_ioc_config(_write(tmp_path, text))

    def test_devices_must_be_list(self, tmp_path) -> None:
        devices_block = """  devices:
    - name: slm
      type: santec_slm200
      params:
        slm_number: 1
        wavelength: 1064
        memory_slots: [3, 4, 5]
"""
        text = VALID_YAML.replace(devices_block, "  devices: slm\n")
        with pytest.raises(IocConfigError, match="devices 必须是列表"):
            load_ioc_config(_write(tmp_path, text))

    def test_device_requires_name_type(self, tmp_path) -> None:
        text = VALID_YAML.replace("    - name: slm\n      type: santec_slm200\n", "    - name: slm\n")
        with pytest.raises(IocConfigError, match="name/type"):
            load_ioc_config(_write(tmp_path, text))
