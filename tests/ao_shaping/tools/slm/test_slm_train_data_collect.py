"""Offline tests for :mod:`ao_shaping.tools.slm.slm_train_data_collect`.

The collector drives real hardware, so everything here runs against fakes or
against the pure planning/sidecar helpers. Nothing opens a device.

The centrepiece is :class:`TestClassifyRoundTrip`: the collector's whole purpose
is to emit records ``ml/hwdataset`` can index, so we drive the **real**
``ml.hwdataset.index._classify`` rather than asserting our own idea of the
contract. That test already caught three defects during development:

* ``_c`` was written with ``n_terms - 1`` entries (piston dropped), and
  ``n_max=4`` therefore produced length 14 -- not triangular -- so **every**
  zernike record was silently dropped as ``ODD_COEFFICIENT_LENGTH``;
* ``_c`` was assigned twice, the second assignment winning;
* ``zernike_panel`` wants ``{(n, m): amp}`` keys while the plan stores Noll
  indices, so the panel was built from keys that could not even be unpacked.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.tools.slm.slm_train_data_collect import (
    CollectParams,
    _effective_store_panel,
    build_plan,
    make_phase,
    sidecar_payload,
    validate_family_prefix,
)
from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms

REPO_ROOT = Path(__file__).resolve().parents[4]


def _params(**overrides) -> CollectParams:
    """A small, fully explicit parameter set; overrides win."""
    base = dict(
        samples=4,
        mode="zernike",
        n_max=4,
        phase_grid=24,
        exposure_ms=(0.4, 1.2, 3.0),
        cam_size=(250, 320),
        subdir_prefix="slm_zernike_shaping",
    )
    base.update(overrides)
    return CollectParams(**base)


class _FakeCamera:
    """Minimal camera: a stable spike so ``measure_spot`` has something to find."""

    def __init__(self, side: int = 32) -> None:
        self._side = side
        self.reads = 0

    def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
        self.reads += 1
        img = np.zeros((self._side, self._side), dtype=np.float64)
        img[self._side // 2, self._side // 2] = 200.0
        return img


class _FakeSlm:
    """Records what was displayed; mirrors the real driver signatures used."""

    def __init__(self) -> None:
        self.displayed: list[tuple[tuple[int, ...], object]] = []

    def create_phase_from_array(
        self, phase_rad: np.ndarray, max_grayscale: int | None = None
    ) -> np.ndarray:
        top = 1023 if max_grayscale is None else int(max_grayscale)
        wrapped = np.asarray(phase_rad, dtype=np.float64) % (2 * np.pi)
        return (wrapped / (2 * np.pi) * top).astype(np.uint16)

    def display_data(self, gray, wait_time_s=None) -> None:
        # The real driver is called WITHOUT memory_number so the firmware rotates
        # the slot itself; assert the collector never passes one.
        self.displayed.append((np.shape(gray), wait_time_s))


class TestPlan:
    """The plan is pure and is where the gap-filling cross-product lives."""

    def test_is_deterministic(self) -> None:
        a = build_plan(_params())
        b = build_plan(_params())
        assert len(a) == len(b) == 4
        assert [p.exposure_ms for p in a] == [p.exposure_ms for p in b]
        assert [p.cam_size for p in a] == [p.cam_size for p in b]
        assert [p.coeffs for p in a] == [p.coeffs for p in b]

    def test_cycles_exposure_and_window_cross_product(self) -> None:
        """One run must spread over the gaps the corpus report found.

        Exposure in the corpus is clumped (1.2 ms alone is 36.5%), so the plan
        deliberately alternates exposure *and* camera window rather than holding
        either fixed -- that is the whole point of the collector.
        """
        plan = build_plan(_params(samples=6))
        assert len({p.exposure_ms for p in plan}) == 3
        assert len({p.cam_size for p in plan}) == 2
        # every (exposure, window) pair appears at least once over 6 points
        pairs = {(p.exposure_ms, p.cam_size) for p in plan}
        assert len(pairs) == 6

    def test_zernike_plan_covers_noll_but_not_piston(self) -> None:
        plan = build_plan(_params(n_max=4))
        coeffs = plan[0].coeffs or {}
        n_terms = calc_n_zernike_terms(4)
        assert n_terms == 15
        assert 1 not in coeffs, "piston is invisible to the detector; do not randomise it"
        assert set(coeffs) == set(range(2, n_terms + 1))

    def test_freeform_plan_carries_grid(self) -> None:
        plan = build_plan(_params(mode="freeform", phase_grid=16))
        assert plan[0].phase_grid == 16

    @pytest.mark.parametrize("prefix", ["slm_pib", "slm_zernike_shaping", "slm_gsnet_square"])
    def test_validate_accepts_known_family_prefixes(self, prefix: str) -> None:
        assert validate_family_prefix(prefix) == prefix

    def test_validate_rejects_unknown_prefix_with_guidance(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            validate_family_prefix("my_campaign")
        message = str(excinfo.value)
        assert "my_campaign" in message
        # must name the recognised set so the caller can pick one
        assert "slm_zernike_shaping" in message


class TestSidecar:
    """The sidecar is what stops the metadata gaps from growing."""

    def _payload(self, **overrides) -> dict:
        params = _params(**overrides)
        return sidecar_payload(params, build_plan(params)[0])

    def test_contains_every_gap_filling_key(self) -> None:
        payload = self._payload()
        for key in (
            "objective",
            "zernike_radius",
            "n_max",
            "cam_size",
            "exposure_ms",
            "exposure_time_ms",
            "cam_type",
            "cam_id",
            "slm_number",
            "slm_wavelength",
            "pupil_center_panel",
            "collector",
            "settle",
        ):
            assert key in payload, f"sidecar lost {key}"

    def test_records_both_exposure_spellings(self) -> None:
        """``index.py`` looks up exp_t -> exposure_ms -> exposure_time_ms."""
        payload = self._payload()
        assert payload["exposure_ms"] == payload["exposure_time_ms"]

    def test_drops_none_values(self) -> None:
        payload = self._payload(pupil_cam_id=None)
        assert "pupil_cam_id" not in payload

    def test_includes_runner_params_block(self) -> None:
        """runner_common reuse: the block must serialise, not silently vanish."""
        payload = self._payload()
        block = payload.get("runner_params")
        assert isinstance(block, dict), "runner_params block missing"
        assert set(block) == {"camera", "slm", "run"}
        # non-default values survive config_payload's default-stripping
        assert block["camera"]["cam_size"] in (250, 320)


class TestPhaseGeneration:
    def test_zernike_panel_is_raw_radians_panel_shape(self) -> None:
        params = _params(n_max=4)
        phase = make_phase(build_plan(params)[0], params)
        assert phase.shape == (1200, 1920)
        assert phase.dtype == np.float32

    def test_freeform_and_turbulence_are_finite(self) -> None:
        for mode in ("freeform", "turbulence"):
            params = _params(mode=mode)
            phase = make_phase(build_plan(params)[0], params)
            assert np.all(np.isfinite(phase)), f"{mode} produced non-finite phase"


class TestEffectiveStorePanel:
    def test_turbulence_forces_panel_on(self) -> None:
        """turbulence has no coefficient vector; without a panel it is dropped."""
        assert _effective_store_panel(_params(mode="turbulence", store_panel=False))

    def test_zernike_respects_the_flag(self) -> None:
        assert not _effective_store_panel(_params(mode="zernike", store_panel=False))
        assert _effective_store_panel(_params(mode="zernike", store_panel=True))


class TestClassifyRoundTrip:
    """Drive the REAL ``ml.hwdataset`` classifier over our records."""

    @staticmethod
    def _record(params: CollectParams):
        from ao_shaping.tools.slm.slm_train_data_collect import acquire

        plan = build_plan(params)
        rows = list(acquire(_FakeCamera(), _FakeSlm(), None, plan, params))
        return rows[0], sidecar_payload(params, plan[0])

    @pytest.mark.parametrize("n_max", [4, 7, 11])
    def test_zernike_c_length_is_triangular_and_kept(self, n_max: int) -> None:
        from ml.hwdataset.index import PhaseSource, _classify

        params = _params(mode="zernike", n_max=n_max)
        row, sidecar = self._record(params)
        assert "_c" in row
        assert len(row["_c"]) == calc_n_zernike_terms(n_max)
        verdict = _classify({**row, "_epoch": 0}, sidecar, "slm_zernike_shaping")
        assert verdict.reason is None, f"record dropped: {verdict.reason}"
        assert verdict.source is PhaseSource.ZERNIKE
        assert verdict.n_max == n_max

    @pytest.mark.parametrize("grid", [16, 24, 32])
    def test_freeform_grid_is_classified_as_freeform(self, grid: int) -> None:
        from ml.hwdataset.index import PhaseSource, _classify

        params = _params(mode="freeform", phase_grid=grid)
        row, sidecar = self._record(params)
        assert len(row["_c"]) == grid * grid
        verdict = _classify({**row, "_epoch": 0}, sidecar, "slm_gsnet_square")
        assert verdict.reason is None
        assert verdict.source is PhaseSource.FREEFORM
        assert verdict.freeform_grid == grid

    def test_turbulence_is_classified_as_panel_gray(self) -> None:
        from ml.hwdataset.index import PhaseSource, _classify

        params = _params(mode="turbulence", store_panel=False)
        row, sidecar = self._record(params)
        assert "_phase" in row, "turbulence must persist a panel even without the flag"
        verdict = _classify({**row, "_epoch": 0}, sidecar, "slm_pib")
        assert verdict.source is PhaseSource.PANEL_GRAY

    def test_record_carries_mandatory_img_and_epoch(self) -> None:
        params = _params()
        row, _ = self._record(params)
        assert "_img" in row and np.asarray(row["_img"]).ndim == 2
        assert "_epoch" in row
        assert "exp_t" in row, "exp_t is index.py's first exposure lookup key"

    def test_never_passes_memory_number_to_display(self) -> None:
        """A repeated slot is a firmware no-op; the kernel handles rotation."""
        params = _params(samples=2)
        slm = _FakeSlm()
        from ao_shaping.tools.slm.slm_train_data_collect import acquire

        list(acquire(_FakeCamera(), slm, None, build_plan(params), params))
        assert slm.displayed, "nothing was displayed"
        for shape, wait in slm.displayed:
            assert wait is None or isinstance(wait, float)


class TestWriterContract:
    """The artefacts must survive the real writer AND the real indexer.

    Regression guard: the collector once passed ``res={"history": [...]}`` -- a
    plain dict -- to :func:`save_recorder_debug_artifacts`, which reads
    ``res.history``. That raised ``AttributeError`` *inside* the flush's
    ``except``, so the run "finished" having written nothing at all. A test that
    only checked record contents could not see it; only writing a real file and
    re-indexing it can.
    """

    def test_written_pkl_is_indexable(self, tmp_path: Path) -> None:
        import json
        import pickle

        from ao_shaping.utils.io.file import Recorder, save_recorder_debug_artifacts
        from ml.hwdataset.index import PhaseSource, _classify

        from ao_shaping.tools.slm.slm_train_data_collect import acquire

        params = _params(samples=3, n_max=4)
        plan = build_plan(params)
        rows = list(acquire(_FakeCamera(), _FakeSlm(), None, plan, params))
        assert rows, "nothing acquired"

        recorder = Recorder(mark="J", mode="max")
        recorder.history = rows
        save_recorder_debug_artifacts(
            res=recorder,
            root_dir=str(tmp_path),
            subdir_prefix="slm_zernike_shaping",
            scalar_keys=("exp_t", "peak", "spot_cx", "spot_cy", "J"),
            objective_keys=(),
            img_keys=("_img", "pupil"),
            d1_keys=("_c",),
            d2_keys=("_phase",),
            json_payload=sidecar_payload(params, plan[0]),
            title="test",
        )

        pkls = list(tmp_path.rglob("*.pkl"))
        sidecars = list(tmp_path.rglob("*.json"))
        assert len(pkls) == 1, "writer produced no pkl"
        assert len(sidecars) == 1

        data = pickle.loads(pkls[0].read_bytes())
        assert sorted(data) == [0, 1, 2]
        row = data[0]
        # keys the writer was NOT told about must be gone; told ones must remain
        assert "_img" in row and "_c" in row
        assert "exp_t" in row and "J" in row

        sidecar = json.loads(sidecars[0].read_text(encoding="utf-8"))
        verdict = _classify(row, sidecar, "slm_zernike_shaping")
        assert verdict.reason is None, f"indexer dropped the record: {verdict.reason}"
        assert verdict.source is PhaseSource.ZERNIKE
        assert verdict.n_max == 4


class TestNoHardware:
    """``--no-hw`` must work on a machine with no SLM/CCD and no SDK."""

    def test_main_no_hw_exits_zero_and_prints_plan(self, capsys) -> None:
        from ao_shaping.tools.slm.slm_train_data_collect import main

        code = main(
            [
                "--no-hw",
                "--samples",
                "3",
                "--mode",
                "zernike",
                "--n-max",
                "4",
                "--exposure-ms",
                "0.4,1.2",
                "--cam-size",
                "250,320",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "zernike" in out
        assert "_c len=15" in out, "plan must state the stored vector length"

    def test_import_does_not_load_a_native_camera_sdk(self) -> None:
        """``--no-hw`` must work on a machine with no camera SDK installed.

        Note what is deliberately *not* asserted: several ``ao_shaping.drivers.*``
        modules do get imported transitively (e.g. ``santec.driver`` via
        ``runner_common``'s ``PANEL_RES``). That is harmless -- they load no
        native library at import time; the DLL/SDK is bound lazily on
        ``open()`` by the PEP-562 lazy contract in ``drivers/_lazy.py``.

        What must stay unloaded is anything that pulls a **native SDK**, because
        those import ``gxipy``/vendor DLLs eagerly and are the reason
        ``tools/slm`` forbids driver imports at module scope.
        """
        code = (
            "import sys, ao_shaping.tools.slm.slm_train_data_collect;"
            "native = [n for n in sys.modules if n in {"
            "'ao_shaping.drivers.ccd.daheng.driver',"
            "'ao_shaping.drivers.ccd.miicam.driver'}];"
            "print('NATIVE=' + repr(native));"
            "print('gxipy=' + repr('gxipy' in sys.modules))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert out.returncode == 0, out.stderr
        assert "NATIVE=[]" in out.stdout, out.stdout
        assert "gxipy=False" in out.stdout, out.stdout