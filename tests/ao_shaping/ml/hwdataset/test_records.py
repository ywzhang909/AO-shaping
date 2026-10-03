"""Tests for :mod:`ml.hwdataset.records`.

Every payload here is **synthetic** and written into ``tmp_path``: the real
corpus is 65.88 GB across 265 pickles, so a test that opened it would be both
slow and non-hermetic. The synthetic panels are ``(40, 60)`` instead of the real
``(1200, 1920)`` -- one thousandth of the pixels, with the same ``(x, y)`` vs
``(row, col)`` convention (``panel_center=(30, 20)`` means column 30, row 20)
so the crop arithmetic is exercised for real rather than skipped.

The expectations are computed by calling
:mod:`ml.hwdataset.transforms` directly. That is deliberate: these tests pin
that :mod:`ml.hwdataset.records` **delegates** the array math instead of
re-deriving it, and that it delegates in the right order (full panel first,
grid reduction second). A hand-rolled expectation would only re-implement the
same bug.
"""

from __future__ import annotations

import ast
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from ml.hwdataset import records as records_module
from ml.hwdataset.index import HwRecordRef, PhaseSource
from ml.hwdataset.records import (
    DEFAULT_PANEL_RESOLUTION,
    HwRecordError,
    HwSample,
    Materialiser,
    MaterialiserConfig,
    PayloadStore,
)
from ml.hwdataset.transforms import (
    farfield_frame_to_grid,
    freeform_grid_to_panel,
    grayscale_to_phase_rad,
    phase_to_grid,
    zernike_coeffs_to_panel,
)

#: Synthetic panel shape as ``(height, width)``; the real panel is ``(1200, 1920)``.
PANEL_SHAPE: tuple[int, int] = (40, 60)
#: Synthetic far-field frame shape; the real frames are ``(250, 248)``.
FRAME_SHAPE: tuple[int, int] = (40, 60)
#: Zernike radial order used by the fixtures: ``(2+1)(2+2)/2 = 6`` terms.
ZERNIKE_N_MAX = 2
ZERNIKE_N_TERMS = 6
#: Freeform cell-grid side used by the fixtures: ``4 * 4 = 16`` cells.
FREEFORM_GRID = 4
FREEFORM_N_TERMS = FREEFORM_GRID * FREEFORM_GRID

#: The fixture configuration required by the spec: a 30x30 crop centred on the
#: panel centre of a 60x40 panel, reduced to 8x8 cells.
CENTER: tuple[int, int] = (30, 20)
RADIUS = 15.0
RESOLUTION: tuple[int, int] = (60, 40)
GRID = 8


class RecorderStandIn:
    """Minimal stand-in for :class:`ao_shaping.utils.io.file.Recorder`.

    Defined at module level because ``pickle`` records the defining module of a
    class, so a class defined inside a test method could not be unpickled back.

    Attributes:
        history: The recorded entries, in recording order.
    """

    def __init__(self, history: list[dict[str, Any]]) -> None:
        """Store the history list.

        Args:
            history: Recorded entries in recording order.
        """
        self.history = history


class BareObject:
    """A pickled object with no ``history`` attribute (module level so it pickles)."""


def _config(**overrides: Any) -> MaterialiserConfig:
    """Build the fixture :class:`MaterialiserConfig`, with optional overrides.

    Args:
        **overrides: Fields to replace on the default fixture configuration.

    Returns:
        The configuration; call ``.validate()`` before using it in a test that
        wants a valid one.
    """
    fields: dict[str, Any] = {
        "grid": GRID,
        "panel_center": CENTER,
        "panel_radius": RADIUS,
        "panel_resolution": RESOLUTION,
        "zernike_radius": 10.0,
        "freeform_radius": 10.0,
    }
    fields.update(overrides)
    return MaterialiserConfig(**fields)


def _panel(value: float = 1.0, shape: tuple[int, int] = PANEL_SHAPE) -> NDArray[np.float32]:
    """Return a small ``float32`` radian panel filled with a constant phase.

    A constant is what makes the coherence assertion exact: a block of identical
    phase has ``hypot(mean(cos), mean(sin)) == 1`` wherever no zero padding is
    mixed in, so a broken block mean shows up as ``contrast != 1`` instead of
    hiding inside a random pattern.

    Args:
        value: The constant phase in radians.
        shape: Panel shape.

    Returns:
        A ``float32`` array shaped like the real ``(1200, 1920)`` panels.
    """
    return np.full(shape, value, dtype=np.float32)


def _gray(value: int = 128, shape: tuple[int, int] = PANEL_SHAPE) -> NDArray[np.uint16]:
    """Return a small ``uint16`` SLM grayscale panel filled with a constant.

    Args:
        value: The constant grayscale level (below the 993 that means 2*pi).
        shape: Panel shape.

    Returns:
        A ``uint16`` array, the dtype the SLM drivers write.
    """
    return np.full(shape, value, dtype=np.uint16)


def _frame(shape: tuple[int, int] = FRAME_SHAPE) -> NDArray[np.uint8]:
    """Return a small ``uint8`` CCD frame with one bright blob.

    The blob matters: :func:`~ml.hwdataset.transforms.farfield_frame_to_grid`
    anchors its window on the **global argmax**, not the frame centre, so a
    featureless or centre-peaked frame would not prove the anchoring at all.

    Args:
        shape: Frame shape.

    Returns:
        A ``uint8`` array shaped like the real ``(250, 248)`` frames.
    """
    frame = np.zeros(shape, dtype=np.uint8)
    row, col = shape[0] // 2, shape[1] // 2
    frame[row - 1 : row + 2, col - 1 : col + 2] = 200
    return frame


def _zernike_coeffs(n_terms: int = ZERNIKE_N_TERMS) -> NDArray[np.float64]:
    """Return a deterministic Noll-order Zernike coefficient vector.

    Args:
        n_terms: Number of coefficients.

    Returns:
        A ``float64`` vector whose values are all distinguishable, so picking
        the wrong record cannot go unnoticed.
    """
    return np.linspace(0.1, 0.6, n_terms, dtype=np.float64)


def _freeform_coeffs(n_terms: int = FREEFORM_N_TERMS) -> NDArray[np.float64]:
    """Return a deterministic flat freeform cell-amplitude vector.

    Args:
        n_terms: Number of cells (``grid * grid``).

    Returns:
        A ``float64`` vector of radians-per-cell.
    """
    return np.linspace(0.0, 1.0, n_terms, dtype=np.float64)


def _dump(root: Path, payload: Any, stem: str) -> Path:
    """Write ``payload`` as a pickle one directory level below ``root``.

    Args:
        root: ``tmp_path``, playing the role of ``data/debug``.
        payload: Any picklable object.
        stem: File stem without the ``.pkl`` suffix.

    Returns:
        The written ``.pkl`` path.
    """
    run_dir = root / "synthetic_run_20260101_120000"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{stem}.pkl"
    path.write_bytes(pickle.dumps(payload))
    return path


def _ref(
    path: Path,
    *,
    position: int = 0,
    key: int | None = 0,
    source: PhaseSource = PhaseSource.PANEL_RAD,
    n_terms: int = 0,
    n_max: int | None = None,
    freeform_grid: int | None = None,
    exposure_ms: float | None = 1.5,
) -> HwRecordRef:
    """Build an :class:`HwRecordRef` with every field the dataclass requires.

    Args:
        path: The pickle the record lives in.
        position: 0-based position inside the payload.
        key: Payload dict key, or ``None`` for a ``history`` payload / non-int key.
        source: Which phase representation the record uses.
        n_terms: ``len(record["_c"])`` when present, else ``0``.
        n_max: Zernike radial order, when known.
        freeform_grid: Freeform cell-grid side, when known.
        exposure_ms: Resolved exposure in milliseconds, or ``None``.

    Returns:
        The reference to hand to :meth:`Materialiser.materialise`.
    """
    return HwRecordRef(
        path=path,
        position=position,
        key=key,
        family="synthetic",
        source=source,
        n_terms=n_terms,
        n_max=n_max,
        freeform_grid=freeform_grid,
        exposure_ms=exposure_ms,
        sidecar={},
    )


def _panel_rad_record(phase_value: float = 1.0) -> dict[str, Any]:
    """Return a ``PANEL_RAD`` record.

    Args:
        phase_value: Constant phase in radians.

    Returns:
        A record mapping with ``_phase`` in radians and a ``_img`` frame.
    """
    return {"_phase": _panel(phase_value), "_img": _frame()}


class TestModuleContract:
    """The importable surface another agent's ``dataset.py`` binds against."""

    def test_all_is_exactly_the_four_contract_names(self) -> None:
        """``__all__`` must be exactly the four names, in order.

        The spec pins this list, so a later "helpful" addition would silently
        widen the contract another module is written against.
        """
        assert records_module.__all__ == [
            "HwRecordError",
            "HwSample",
            "MaterialiserConfig",
            "PayloadStore",
        ]

    def test_materialiser_is_importable_despite_not_being_in_all(self) -> None:
        """``Materialiser`` is public but deliberately omitted from ``__all__``.

        ``__all__`` only restricts ``from ... import *``, so the class must
        still be reachable by name -- that is how ``dataset.py`` will use it.
        """
        assert Materialiser is records_module.Materialiser
        assert "Materialiser" not in records_module.__all__

    def test_module_docstring_precedes_future_import(self) -> None:
        """``__doc__`` must be the real docstring, not ``None``.

        Getting ``from __future__ import annotations`` above the docstring
        silently sets ``__doc__`` to ``None``; this repo's convention is
        docstring first.
        """
        assert records_module.__doc__ is not None
        assert "hardware record" in records_module.__doc__

    def test_module_does_not_import_torch(self) -> None:
        """The module under test must stay torch-free.

        ``transforms`` uses torch internally, but importing it here must not
        make ``records`` a torch-dependent module for the Dataset.

        This walks the AST rather than grepping for the text ``"import torch"``,
        because that substring misses ``from torch import ...`` and
        ``import torch.nn``-style spellings that a real regression would use.
        Every ``import``/``from`` target in the module is checked instead.
        """
        tree = ast.parse(Path(records_module.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
        assert "torch" not in imported, f"records must not import torch, got {sorted(imported)}"
        # The test is only meaningful if the import scan actually works.
        assert "numpy" in imported and "loguru" in imported


class TestConfigDefaultsAndValidation:
    """``MaterialiserConfig`` defaults and its field-naming validation."""

    def test_documented_defaults(self) -> None:
        """The dataclass defaults must match the documented contract."""
        config = MaterialiserConfig()
        assert config.grid == 64
        assert config.panel_resolution == DEFAULT_PANEL_RESOLUTION == (1920, 1200)
        assert config.panel_center == (960, 600)
        assert config.panel_radius == 500.0
        assert config.image_mode == "abs255"
        assert config.zernike_radius == 300.0
        assert config.freeform_radius == 450.0

    def test_fixture_config_validates(self) -> None:
        """The fixture configuration used by every other test must be valid."""
        _config().validate()

    @pytest.mark.parametrize(
        ("field", "value", "needle"),
        [
            ("grid", 0, "grid"),
            ("grid", -3, "grid"),
            ("slm_max_gray", 0, "slm_max_gray"),
            ("image_mode", "nope", "image_mode"),
            ("panel_center", (30,), "panel_center"),
            ("panel_center", (float("nan"), 20), "panel_center"),
            ("panel_radius", 0.0, "panel_radius"),
            ("panel_radius", -10.0, "panel_radius"),
            ("panel_resolution", (60,), "panel_resolution"),
            ("panel_resolution", (60, 0), "panel_resolution"),
            ("zernike_radius", 0.5, "zernike_radius"),
            ("freeform_radius", 0.0, "freeform_radius"),
        ],
    )
    def test_invalid_field_raises_value_error_naming_it(
        self, field: str, value: Any, needle: str
    ) -> None:
        """Every bad field must raise ``ValueError`` naming that field.

        The message has to name the config field the operator must change, not
        an array shape from deep inside a transform.
        """
        config = _config(**{field: value})
        with pytest.raises(ValueError, match=needle):
            config.validate()

    def test_constructor_validates_and_propagates(self) -> None:
        """``Materialiser(config)`` must validate eagerly, not at first use."""
        with pytest.raises(ValueError, match="grid"):
            Materialiser(_config(grid=0))

    def test_default_config_when_none(self) -> None:
        """``Materialiser()`` must build a usable default configuration."""
        materialiser = Materialiser()
        assert materialiser.config == MaterialiserConfig()
        assert materialiser.store.capacity == 1

    def test_cache_size_is_not_part_of_the_config(self) -> None:
        """The payload cache must not leak into the cache key.

        ``cache_size`` changes RAM and speed, never the numbers, so it belongs
        to the constructor rather than to :class:`MaterialiserConfig`.
        """
        assert not hasattr(MaterialiserConfig(), "cache_size")
        assert Materialiser(_config(), cache_size=7).config == _config()


class TestPhaseSources:
    """All four phase representations, each checked against ``transforms``."""

    def test_panel_rad_uses_phase_field_verbatim(self, tmp_path: Path) -> None:
        """``PANEL_RAD``: ``_phase`` is already radians, used as-is.

        No ``mod 2*pi`` and no scaling may be applied -- the driver is the only
        wrap point in this repo.
        """
        phase = _panel(1.25)
        path = _dump(tmp_path, {0: {"_phase": phase, "_img": _frame()}}, "panel_rad")
        sample = Materialiser(_config()).materialise(_ref(path))

        expected_cos, expected_sin = phase_to_grid(
            phase, GRID, center=CENTER, radius=RADIUS
        )
        np.testing.assert_allclose(sample.phase_cos, expected_cos, atol=1e-6)
        np.testing.assert_allclose(sample.phase_sin, expected_sin, atol=1e-6)

    def test_panel_gray_converts_through_the_canonical_transform(
        self, tmp_path: Path
    ) -> None:
        """``PANEL_GRAY``: ``_phase`` is grayscale, converted with ``slm_max_gray``."""
        gray = _gray(300)
        config = _config()
        path = _dump(tmp_path, {0: {"_phase": gray, "_img": _frame()}}, "panel_gray")
        sample = Materialiser(config).materialise(_ref(path, source=PhaseSource.PANEL_GRAY))

        radians = grayscale_to_phase_rad(gray, config.slm_max_gray)
        expected_cos, expected_sin = phase_to_grid(
            radians, GRID, center=CENTER, radius=RADIUS
        )
        np.testing.assert_allclose(sample.phase_cos, expected_cos, atol=1e-6)
        np.testing.assert_allclose(sample.phase_sin, expected_sin, atol=1e-6)

    def test_zernike_expands_before_reducing(self, tmp_path: Path) -> None:
        """``ZERNIKE``: coefficients expand to a panel, *then* the grid reduces.

        The order is the whole point: reducing a coefficient vector directly
        would be meaningless, and this pins that ``records`` calls
        ``zernike_coeffs_to_panel`` then ``phase_to_grid`` in that order.
        """
        coeffs = _zernike_coeffs()
        config = _config()
        path = _dump(tmp_path, {0: {"_c": coeffs, "_img": _frame()}}, "zernike")
        ref = _ref(
            path,
            source=PhaseSource.ZERNIKE,
            n_terms=ZERNIKE_N_TERMS,
            n_max=ZERNIKE_N_MAX,
        )
        sample = Materialiser(config).materialise(ref)

        panel = zernike_coeffs_to_panel(
            coeffs,
            n_max=ZERNIKE_N_MAX,
            resolution=RESOLUTION,
            radius=config.zernike_radius,
        )
        expected_cos, expected_sin = phase_to_grid(
            panel, GRID, center=CENTER, radius=RADIUS
        )
        np.testing.assert_allclose(sample.phase_cos, expected_cos, atol=1e-6)
        np.testing.assert_allclose(sample.phase_sin, expected_sin, atol=1e-6)

    def test_zernike_uses_the_configured_radius(self, tmp_path: Path) -> None:
        """A different ``zernike_radius`` must give different numbers.

        Guards against the radius being hard-coded instead of read from the
        config, which is what makes the on-disk cache key meaningful.
        """
        coeffs = _zernike_coeffs()
        path = _dump(tmp_path, {0: {"_c": coeffs, "_img": _frame()}}, "zernike_r")
        ref = _ref(
            path,
            source=PhaseSource.ZERNIKE,
            n_terms=ZERNIKE_N_TERMS,
            n_max=ZERNIKE_N_MAX,
        )
        near = Materialiser(_config(zernike_radius=10.0)).materialise(ref)
        wide = Materialiser(_config(zernike_radius=20.0)).materialise(ref)
        assert not np.allclose(near.phase_cos, wide.phase_cos)

    def test_freeform_upsamples_before_reducing(self, tmp_path: Path) -> None:
        """``FREEFORM``: ``grid*grid`` cells upsample to a panel, then reduce."""
        coeffs = _freeform_coeffs()
        config = _config()
        path = _dump(tmp_path, {0: {"_c": coeffs, "_img": _frame()}}, "freeform")
        ref = _ref(
            path,
            source=PhaseSource.FREEFORM,
            n_terms=FREEFORM_N_TERMS,
            freeform_grid=FREEFORM_GRID,
        )
        sample = Materialiser(config).materialise(ref)

        panel = freeform_grid_to_panel(
            coeffs, grid=FREEFORM_GRID, resolution=RESOLUTION
        )
        expected_cos, expected_sin = phase_to_grid(
            panel, GRID, center=CENTER, radius=RADIUS
        )
        np.testing.assert_allclose(sample.phase_cos, expected_cos, atol=1e-6)
        np.testing.assert_allclose(sample.phase_sin, expected_sin, atol=1e-6)

    def test_image_delegates_to_the_canonical_transform(self, tmp_path: Path) -> None:
        """``_img`` must go through ``farfield_frame_to_grid`` with ``image_mode``."""
        frame = _frame()
        config = _config()
        path = _dump(tmp_path, {0: {"_phase": _panel(), "_img": frame}}, "image")
        sample = Materialiser(config).materialise(_ref(path))
        expected = farfield_frame_to_grid(frame, GRID, mode=config.image_mode)
        np.testing.assert_allclose(sample.image, expected, atol=1e-6)

    @pytest.mark.parametrize("mode", ["abs255", "peak", "raw"])
    def test_every_image_mode_is_honoured(self, tmp_path: Path, mode: str) -> None:
        """Each documented ``image_mode`` must be passed through unchanged."""
        frame = _frame()
        config = _config(image_mode=mode)
        path = _dump(tmp_path, {0: {"_phase": _panel(), "_img": frame}}, f"mode_{mode}")
        sample = Materialiser(config).materialise(_ref(path))
        expected = farfield_frame_to_grid(frame, GRID, mode=mode)
        np.testing.assert_allclose(sample.image, expected, atol=1e-6)

    def test_abs255_keeps_absolute_intensity(self, tmp_path: Path) -> None:
        """A dimmer frame must give a dimmer image -- the reason ``abs255`` is default.

        The exposure time is a model *input*, so peak-normalising the target
        would destroy the brightness that encodes it.
        """
        bright = _frame()
        dim = (bright // 4).astype(np.uint8)
        config = _config()
        path_a = _dump(tmp_path, {0: {"_phase": _panel(), "_img": bright}}, "bright")
        path_b = _dump(tmp_path, {0: {"_phase": _panel(), "_img": dim}}, "dim")
        first = Materialiser(config).materialise(_ref(path_a))
        second = Materialiser(config).materialise(_ref(path_b))
        assert first.image.sum() > second.image.sum()


class TestContrast:
    """``contrast`` must be exactly the per-cell phasor magnitude."""

    def test_contrast_equals_hypot_of_the_two_channels(self, tmp_path: Path) -> None:
        """``contrast == hypot(phase_cos, phase_sin)``, elementwise."""
        path = _dump(tmp_path, {0: _panel_rad_record(0.7)}, "contrast")
        sample = Materialiser(_config()).materialise(_ref(path))
        np.testing.assert_allclose(
            sample.contrast,
            np.hypot(sample.phase_cos, sample.phase_sin),
            atol=1e-6,
        )

    def test_contrast_is_bounded_by_one(self, tmp_path: Path) -> None:
        """Coherence lives in ``[0, 1]``; a larger value would be a broken mean."""
        path = _dump(tmp_path, {0: _panel_rad_record(2.9)}, "bounds")
        sample = Materialiser(_config()).materialise(_ref(path))
        assert sample.contrast.min() >= 0.0
        assert sample.contrast.max() <= 1.0 + 1e-6

    def test_uniform_phase_is_fully_coherent_at_grid_one(self, tmp_path: Path) -> None:
        """A constant phase gives ``contrast == 1`` where no padding is mixed in.

        At ``grid=1`` the whole 30x30 crop is one cell, so the zero-pad-up rule
        adds nothing and the assertion is exact -- this is the sharpest check
        that the coherent (complex-phasor) mean is used rather than an
        ``arctan2`` of an arithmetic mean.
        """
        path = _dump(tmp_path, {0: _panel_rad_record(1.0)}, "uniform")
        sample = Materialiser(_config(grid=1)).materialise(_ref(path))
        assert sample.phase_cos.shape == (1, 1)
        np.testing.assert_allclose(sample.contrast, [[1.0]], atol=1e-6)

    def test_contrast_is_not_stored_in_the_record(self, tmp_path: Path) -> None:
        """``contrast`` is derived, never read from a ``_contrast`` field."""
        record = _panel_rad_record(0.3)
        record["_contrast"] = np.zeros((GRID, GRID), dtype=np.float32)
        path = _dump(tmp_path, {0: record}, "fake_contrast")
        sample = Materialiser(_config()).materialise(_ref(path))
        assert sample.contrast.max() > 0.5


class TestExposure:
    """``exposure_ms`` comes from the ref, not re-derived from the record."""

    @pytest.mark.parametrize("exposure", [0.0, 0.8, 1.5, 40.0, None])
    def test_exposure_is_propagated_verbatim(
        self, tmp_path: Path, exposure: float | None
    ) -> None:
        """Whatever the index resolved is what the sample carries."""
        path = _dump(tmp_path, {0: _panel_rad_record()}, f"exp_{exposure}")
        ref = _ref(path, exposure_ms=exposure)
        assert Materialiser(_config()).materialise(ref).exposure_ms == exposure

    def test_record_exp_t_does_not_override_the_ref(self, tmp_path: Path) -> None:
        """The index already applied record-then-sidecar precedence.

        Re-deriving it here would silently reverse a deliberate decision.
        """
        record = _panel_rad_record()
        record["exp_t"] = 99.0
        path = _dump(tmp_path, {0: record}, "exp_t")
        sample = Materialiser(_config()).materialise(_ref(path, exposure_ms=1.5))
        assert sample.exposure_ms == 1.5


class TestStatelessness:
    """Two materialisers must agree bit for bit."""

    def test_repeated_calls_are_identical(self, tmp_path: Path) -> None:
        """The same ref through the same instance must give the same arrays."""
        path = _dump(tmp_path, {0: _panel_rad_record(1.1)}, "repeat")
        materialiser = Materialiser(_config())
        ref = _ref(path)
        first = materialiser.materialise(ref)
        second = materialiser.materialise(ref)
        for name in ("phase_cos", "phase_sin", "image", "contrast"):
            np.testing.assert_array_equal(
                getattr(first, name), getattr(second, name), err_msg=name
            )

    def test_two_instances_agree(self, tmp_path: Path) -> None:
        """A cold instance must match a warm one -- no hidden state."""
        payload = {0: _panel_rad_record(1.1), 1: _panel_rad_record(2.2)}
        path = _dump(tmp_path, payload, "two_instances")
        ref = _ref(path)
        cold = Materialiser(_config()).materialise(ref)
        warm = Materialiser(_config())
        warm.materialise(ref)
        warm.materialise(_ref(path, position=1, key=1))
        for name in ("phase_cos", "phase_sin", "image", "contrast"):
            np.testing.assert_array_equal(
                getattr(cold, name), getattr(warm.materialise(ref), name), err_msg=name
            )

    def test_a_different_grid_changes_the_output(self, tmp_path: Path) -> None:
        """``grid`` must reach the transforms, not be ignored."""
        path = _dump(tmp_path, {0: _panel_rad_record(1.1)}, "grid_effect")
        ref = _ref(path)
        eight = Materialiser(_config(grid=8)).materialise(ref)
        sixteen = Materialiser(_config(grid=16)).materialise(ref)
        assert eight.phase_cos.shape == (8, 8)
        assert sixteen.phase_cos.shape == (16, 16)

    def test_grid_one_is_allowed(self, tmp_path: Path) -> None:
        """``grid=1`` is the degenerate but legal 1x1 sample."""
        path = _dump(tmp_path, {0: _panel_rad_record(0.5)}, "grid_one")
        sample = Materialiser(_config(grid=1)).materialise(_ref(path))
        assert sample.phase_cos.shape == (1, 1)
        assert sample.phase_sin.shape == (1, 1)
        assert sample.image.shape == (1, 1)
        assert sample.contrast.shape == (1, 1)


class TestPayloadStoreLru:
    """Bounded LRU over payloads, keyed by resolved path."""

    def test_capacity_zero_never_retains(self, tmp_path: Path) -> None:
        """``capacity=0`` disables retention but still serves records."""
        path = _dump(tmp_path, {0: _panel_rad_record()}, "cap0")
        store = PayloadStore(capacity=0)
        materialiser = Materialiser(_config(), cache_size=0)
        assert len(store) == 0
        materialiser.materialise(_ref(path))
        assert len(materialiser.store) == 0
        assert materialiser.store.cached_paths == ()

    def test_negative_capacity_rejected(self) -> None:
        """A negative capacity is a mistake, not a silent no-op."""
        with pytest.raises(ValueError, match="capacity"):
            PayloadStore(capacity=-1)

    def test_cached_paths_are_lru_ordered(self, tmp_path: Path) -> None:
        """``cached_paths`` runs least- to most-recently used."""
        store = PayloadStore(capacity=3)
        paths = [_dump(tmp_path, {0: _panel_rad_record()}, f"lru{i}") for i in range(3)]
        for path in paths:
            store.get(_ref(path))
        assert store.cached_paths == tuple(p.resolve() for p in paths)
        assert len(store) == 3

    def test_eviction_happens_at_capacity(self, tmp_path: Path) -> None:
        """A fourth file must evict the least recently used one."""
        store = PayloadStore(capacity=2)
        paths = [_dump(tmp_path, {0: _panel_rad_record()}, f"evict{i}") for i in range(3)]
        store.get(_ref(paths[0]))
        store.get(_ref(paths[1]))
        store.get(_ref(paths[2]))
        assert store.cached_paths == (paths[1].resolve(), paths[2].resolve())

    def test_reaccess_refreshes_recency(self, tmp_path: Path) -> None:
        """Touching an old entry must save it from the next eviction."""
        store = PayloadStore(capacity=2)
        paths = [_dump(tmp_path, {0: _panel_rad_record()}, f"touch{i}") for i in range(3)]
        store.get(_ref(paths[0]))
        store.get(_ref(paths[1]))
        store.get(_ref(paths[0]))  # paths[0] is now the most recently used
        store.get(_ref(paths[2]))  # must evict paths[1]
        assert store.cached_paths == (paths[0].resolve(), paths[2].resolve())

    def test_relative_and_absolute_paths_share_one_entry(
        self, tmp_path: Path
    ) -> None:
        """The cache key is the resolved path, so one file is one entry."""
        path = _dump(tmp_path, {0: _panel_rad_record()}, "resolve")
        store = PayloadStore(capacity=4)
        store.get(_ref(path))
        store.get(_ref(Path(path).absolute()))
        assert len(store) == 1

    def test_clear_empties_the_cache(self, tmp_path: Path) -> None:
        """``clear`` must release every payload."""
        path = _dump(tmp_path, {0: _panel_rad_record()}, "clear")
        materialiser = Materialiser(_config())
        materialiser.materialise(_ref(path))
        assert len(materialiser.store) == 1
        materialiser.clear()
        assert len(materialiser.store) == 0
        assert materialiser.store.cached_paths == ()

    def test_clear_still_allows_further_reads(self, tmp_path: Path) -> None:
        """Clearing is a memory operation, not a teardown."""
        path = _dump(tmp_path, {0: _panel_rad_record()}, "clear_again")
        materialiser = Materialiser(_config())
        materialiser.materialise(_ref(path))
        materialiser.clear()
        materialiser.materialise(_ref(path))
        assert len(materialiser.store) == 1


class TestPickling:
    """Windows ``spawn`` re-pickles the Dataset for every worker."""

    def test_getstate_drops_the_payloads(self, tmp_path: Path) -> None:
        """Only the capacity may be in the state; payloads would be gigabytes."""
        path = _dump(tmp_path, {0: _panel_rad_record()}, "state")
        store = PayloadStore(capacity=4)
        store.get(_ref(path))
        assert len(store) == 1
        assert store.__getstate__() == {"capacity": 4}

    def test_store_roundtrip_is_empty_but_usable(self, tmp_path: Path) -> None:
        """A restored store keeps its capacity, drops its cache, still reads."""
        path = _dump(tmp_path, {0: _panel_rad_record()}, "roundtrip")
        store = PayloadStore(capacity=3)
        store.get(_ref(path))
        restored = pickle.loads(pickle.dumps(store))
        assert restored.capacity == 3
        assert len(restored) == 0
        assert restored.get(_ref(path))["_img"] is not None

    def test_materialiser_roundtrip_is_empty_but_usable(self, tmp_path: Path) -> None:
        """The whole materialiser must survive a pickle for DataLoader workers."""
        path = _dump(tmp_path, {0: _panel_rad_record(0.9)}, "mat_roundtrip")
        materialiser = Materialiser(_config(), cache_size=2)
        materialiser.materialise(_ref(path))
        restored = pickle.loads(pickle.dumps(materialiser))
        assert restored.config == _config()
        assert len(restored.store) == 0
        assert restored.materialise(_ref(path)).contrast.max() > 0.0

    def test_restored_capacity_matches(self) -> None:
        """``__setstate__`` must restore the capacity, not default it to 1."""
        restored = pickle.loads(pickle.dumps(PayloadStore(capacity=9)))
        assert restored.capacity == 9


class TestPayloadShapes:
    """Both measured payload shapes, and the positional fallback."""

    def test_dict_payload_addressed_by_key(self, tmp_path: Path) -> None:
        """``dict[int, dict]`` is addressed by ``ref.key``."""
        payload = {i: _panel_rad_record(0.1 * (i + 1)) for i in range(3)}
        path = _dump(tmp_path, payload, "by_key")
        materialiser = Materialiser(_config(grid=1))
        for i in (0, 1, 2):
            sample = materialiser.materialise(_ref(path, position=i, key=i))
            expected = phase_to_grid(
                _panel(0.1 * (i + 1)), 1, center=CENTER, radius=RADIUS
            )
            np.testing.assert_allclose(sample.phase_cos, expected[0], atol=1e-6)

    def test_recorder_payload_addressed_by_position(self, tmp_path: Path) -> None:
        """An object exposing ``.history`` is addressed by ``ref.position``."""
        payload = RecorderStandIn([_panel_rad_record(0.2), _panel_rad_record(0.8)])
        path = _dump(tmp_path, payload, "recorder")
        materialiser = Materialiser(_config(grid=1))
        for position, value in ((0, 0.2), (1, 0.8)):
            ref = _ref(path, position=position, key=None)
            sample = materialiser.materialise(ref)
            expected = phase_to_grid(_panel(value), 1, center=CENTER, radius=RADIUS)
            np.testing.assert_allclose(sample.phase_cos, expected[0], atol=1e-6)

    def test_non_int_key_falls_back_to_sorted_position(self, tmp_path: Path) -> None:
        """A ``key=None`` ref must use the index's sorted order, not insertion order.

        ``ml.hwdataset.index._dict_sort_key`` sorts integer keys numerically first
        and everything else by its string form; ``records`` mirrors that exactly.
        Getting it wrong would read a *valid but wrong* record -- the quietest
        possible failure.
        """
        payload = {"b": _panel_rad_record(0.5), "a": _panel_rad_record(2.0)}
        path = _dump(tmp_path, payload, "str_keys")
        materialiser = Materialiser(_config(grid=1))
        # sorted(str): "a" then "b" -> position 0 is the 2.0 rad record.
        first = materialiser.materialise(_ref(path, position=0, key=None))
        expected_first = phase_to_grid(_panel(2.0), 1, center=CENTER, radius=RADIUS)
        np.testing.assert_allclose(first.phase_cos, expected_first[0], atol=1e-6)

        second = materialiser.materialise(_ref(path, position=1, key=None))
        expected_second = phase_to_grid(_panel(0.5), 1, center=CENTER, radius=RADIUS)
        np.testing.assert_allclose(second.phase_cos, expected_second[0], atol=1e-6)

    def test_integer_keys_sort_numerically_not_lexically(self, tmp_path: Path) -> None:
        """Keys ``0, 2, 10`` must order as ``0, 2, 10`` -- not ``0, 10, 2``."""
        payload = {
            0: _panel_rad_record(0.1),
            2: _panel_rad_record(0.2),
            10: _panel_rad_record(0.3),
        }
        path = _dump(tmp_path, payload, "int_keys")
        materialiser = Materialiser(_config(grid=1))
        for position, value in ((0, 0.1), (1, 0.2), (2, 0.3)):
            sample = materialiser.materialise(_ref(path, position=position, key=None))
            expected = phase_to_grid(_panel(value), 1, center=CENTER, radius=RADIUS)
            np.testing.assert_allclose(sample.phase_cos, expected[0], atol=1e-6)

    def test_get_returns_the_mapping_not_the_payload(self, tmp_path: Path) -> None:
        """``get`` must hand back one record so a caller cannot retain a whole file."""
        path = _dump(tmp_path, {0: _panel_rad_record()}, "mapping")
        store = PayloadStore()
        record = store.get(_ref(path))
        assert set(record) == {"_phase", "_img"}
        assert isinstance(record["_img"], np.ndarray)

    def test_bare_object_without_history_is_rejected(self, tmp_path: Path) -> None:
        """A payload that is neither shape must be named, not silently skipped."""
        path = _dump(tmp_path, BareObject(), "bare")
        with pytest.raises(HwRecordError, match="history"):
            PayloadStore().get(_ref(path, key=None))


class TestErrorCases:
    """Every specified failure must raise ``HwRecordError``, loudly."""

    def test_missing_phase_key(self, tmp_path: Path) -> None:
        """No ``_phase`` on a panel source is an error, not a flat phase."""
        path = _dump(tmp_path, {0: {"_img": _frame()}}, "no_phase")
        with pytest.raises(HwRecordError, match="_phase"):
            Materialiser(_config()).materialise(_ref(path))

    def test_missing_coeff_key(self, tmp_path: Path) -> None:
        """No ``_c`` on a Zernike source is an error."""
        path = _dump(tmp_path, {0: {"_img": _frame()}}, "no_c")
        ref = _ref(
            path, source=PhaseSource.ZERNIKE, n_terms=ZERNIKE_N_TERMS, n_max=ZERNIKE_N_MAX
        )
        with pytest.raises(HwRecordError, match="_c"):
            Materialiser(_config()).materialise(ref)

    def test_missing_image_key(self, tmp_path: Path) -> None:
        """No ``_img`` is an error; there is no such thing as an absent target."""
        path = _dump(tmp_path, {0: {"_phase": _panel()}}, "no_img")
        with pytest.raises(HwRecordError, match="_img"):
            Materialiser(_config()).materialise(_ref(path))

    def test_zernike_without_n_max(self, tmp_path: Path) -> None:
        """A Zernike ref with no ``n_max`` cannot fix the mode count."""
        path = _dump(tmp_path, {0: {"_c": _zernike_coeffs(), "_img": _frame()}}, "no_nmax")
        ref = _ref(path, source=PhaseSource.ZERNIKE, n_terms=ZERNIKE_N_TERMS)
        with pytest.raises(HwRecordError, match="n_max"):
            Materialiser(_config()).materialise(ref)

    def test_freeform_without_grid(self, tmp_path: Path) -> None:
        """A freeform ref with no ``freeform_grid`` cannot fix the cell count."""
        path = _dump(
            tmp_path, {0: {"_c": _freeform_coeffs(), "_img": _frame()}}, "no_grid"
        )
        ref = _ref(path, source=PhaseSource.FREEFORM, n_terms=FREEFORM_N_TERMS)
        with pytest.raises(HwRecordError, match="freeform_grid"):
            Materialiser(_config()).materialise(ref)

    def test_non_finite_phase(self, tmp_path: Path) -> None:
        """A NaN phase would be zeroed by the transform, fabricating a flat sample."""
        phase = _panel()
        phase[5, 5] = np.nan
        path = _dump(tmp_path, {0: {"_phase": phase, "_img": _frame()}}, "nan_phase")
        with pytest.raises(HwRecordError, match="non-finite"):
            Materialiser(_config()).materialise(_ref(path))

    def test_non_finite_infinite_coeffs(self, tmp_path: Path) -> None:
        """An infinite coefficient is just as unusable as a NaN."""
        coeffs = _zernike_coeffs()
        coeffs[2] = np.inf
        path = _dump(tmp_path, {0: {"_c": coeffs, "_img": _frame()}}, "inf_c")
        ref = _ref(
            path, source=PhaseSource.ZERNIKE, n_terms=ZERNIKE_N_TERMS, n_max=ZERNIKE_N_MAX
        )
        with pytest.raises(HwRecordError, match="non-finite"):
            Materialiser(_config()).materialise(ref)

    def test_non_finite_image(self, tmp_path: Path) -> None:
        """A NaN target pixel would become a hard zero in the window."""
        frame = _frame().astype(np.float32)
        frame[3, 3] = np.nan
        path = _dump(
            tmp_path, {0: {"_phase": _panel(), "_img": frame}}, "nan_img"
        )
        with pytest.raises(HwRecordError, match="non-finite"):
            Materialiser(_config()).materialise(_ref(path))

    def test_unpicklable_file(self, tmp_path: Path) -> None:
        """A corrupt dump must name the file so an operator can delete it."""
        path = tmp_path / "corrupt.pkl"
        path.write_bytes(b"\x80\x04not-a-pickle")
        with pytest.raises(HwRecordError, match="Could not unpickle"):
            Materialiser(_config()).materialise(_ref(path))

    def test_missing_file(self, tmp_path: Path) -> None:
        """A vanished file is an error, not an empty sample."""
        with pytest.raises(HwRecordError, match="Could not unpickle"):
            Materialiser(_config()).materialise(_ref(tmp_path / "nope.pkl"))

    def test_none_payload_from_aborted_run(self, tmp_path: Path) -> None:
        """3 of the 265 corpus pickles unpickle to ``None``."""
        path = _dump(tmp_path, None, "aborted")
        with pytest.raises(HwRecordError, match="neither a dict"):
            Materialiser(_config()).materialise(_ref(path, key=None))

    def test_absent_key(self, tmp_path: Path) -> None:
        """A ``key`` not in the payload must be named."""
        path = _dump(tmp_path, {0: _panel_rad_record()}, "absent_key")
        with pytest.raises(HwRecordError, match="absent"):
            Materialiser(_config()).materialise(_ref(path, key=99))

    def test_dict_position_out_of_range(self, tmp_path: Path) -> None:
        """A positional ref past the end of a dict payload is an error."""
        path = _dump(tmp_path, {0: _panel_rad_record()}, "range_dict")
        with pytest.raises(HwRecordError, match="out of range"):
            Materialiser(_config()).materialise(_ref(path, position=5, key=None))

    def test_history_position_out_of_range(self, tmp_path: Path) -> None:
        """The same for a ``history`` payload."""
        path = _dump(tmp_path, RecorderStandIn([_panel_rad_record()]), "range_hist")
        with pytest.raises(HwRecordError, match="out of range"):
            Materialiser(_config()).materialise(_ref(path, position=5, key=None))

    def test_record_is_not_a_mapping(self, tmp_path: Path) -> None:
        """A payload entry that is not a record mapping must be named."""
        path = _dump(tmp_path, {0: 42}, "not_mapping")
        with pytest.raises(HwRecordError, match="not a record mapping"):
            Materialiser(_config()).materialise(_ref(path))

    def test_zernike_coeff_length_disagrees_with_n_max(self, tmp_path: Path) -> None:
        """A transform ``ValueError`` must surface as ``HwRecordError``."""
        path = _dump(
            tmp_path, {0: {"_c": _zernike_coeffs(10), "_img": _frame()}}, "bad_zernike"
        )
        ref = _ref(
            path, source=PhaseSource.ZERNIKE, n_terms=10, n_max=ZERNIKE_N_MAX
        )
        with pytest.raises(HwRecordError, match="Could not build the panel phase"):
            Materialiser(_config()).materialise(ref)

    def test_freeform_length_not_a_square(self, tmp_path: Path) -> None:
        """A freeform vector whose length is not ``grid*grid`` is an error."""
        path = _dump(
            tmp_path, {0: {"_c": np.zeros(15), "_img": _frame()}}, "bad_freeform"
        )
        ref = _ref(path, source=PhaseSource.FREEFORM, n_terms=15, freeform_grid=4)
        with pytest.raises(HwRecordError, match="Could not build the panel phase"):
            Materialiser(_config()).materialise(ref)

    def test_panel_not_two_dimensional(self, tmp_path: Path) -> None:
        """A 1-D ``_phase`` must be rejected by the crop, and named."""
        path = _dump(
            tmp_path, {0: {"_phase": np.zeros(8, np.float32), "_img": _frame()}}, "1d"
        )
        with pytest.raises(HwRecordError, match="Could not reduce the phase"):
            Materialiser(_config()).materialise(_ref(path))

    def test_image_not_two_dimensional(self, tmp_path: Path) -> None:
        """A 1-D ``_img`` must be rejected and named."""
        path = _dump(
            tmp_path, {0: {"_phase": _panel(), "_img": np.zeros(8, np.uint8)}}, "img_1d"
        )
        with pytest.raises(HwRecordError, match="Could not reduce the far-field"):
            Materialiser(_config()).materialise(_ref(path))

    def test_error_message_always_names_path_and_position(self, tmp_path: Path) -> None:
        """A 265-file corpus gives no other way to find the offending record."""
        path = _dump(tmp_path, {0: {"_img": _frame()}}, "where")
        ref = _ref(path, position=3)
        with pytest.raises(HwRecordError) as caught:
            Materialiser(_config()).materialise(ref)
        message = str(caught.value)
        assert path.name in message
        assert "position 3" in message

    def test_phase_grid_raises_the_same_errors(self, tmp_path: Path) -> None:
        """``phase_grid`` must be as strict as ``materialise``.

        It is a public entry point, so it cannot be the weaker one.
        """
        path = _dump(tmp_path, {0: {"_img": _frame()}}, "phase_grid_err")
        with pytest.raises(HwRecordError, match="_phase"):
            Materialiser(_config()).phase_grid(_ref(path))


class TestOwnershipAndPurity:
    """The payload is shared; the outputs must not be."""

    def test_outputs_own_their_data(self, tmp_path: Path) -> None:
        """Every returned array must own a C-contiguous ``float32`` buffer."""
        path = _dump(tmp_path, {0: _panel_rad_record()}, "ownership")
        sample = Materialiser(_config()).materialise(_ref(path))
        for name in ("phase_cos", "phase_sin", "image", "contrast"):
            array = getattr(sample, name)
            assert array.dtype == np.float32, name
            assert array.flags.owndata, name
            assert array.flags.c_contiguous, name

    def test_two_samples_do_not_alias(self, tmp_path: Path) -> None:
        """Two records from one file must not share memory.

        The payload is held by the LRU, so an aliasing sample would let a caller
        normalising in place corrupt a *different* record. Mutating through
        ``[:]`` rather than ``*=`` because ``HwSample`` is a frozen dataclass:
        ``x.field *= k`` still rebinds the attribute and would raise.
        """
        payload = {0: _panel_rad_record(0.4), 1: _panel_rad_record(0.6)}
        path = _dump(tmp_path, payload, "alias")
        materialiser = Materialiser(_config())
        first = materialiser.materialise(_ref(path, position=0, key=0))
        second = materialiser.materialise(_ref(path, position=1, key=1))

        second_before = second.phase_cos.copy()
        first.phase_cos[:] = -first.phase_cos
        np.testing.assert_array_equal(second.phase_cos, second_before)
        assert not np.allclose(first.phase_cos, second.phase_cos)
        assert not np.shares_memory(first.phase_cos, second.phase_cos)

    def test_mutating_an_output_does_not_corrupt_the_next_read(
        self, tmp_path: Path
    ) -> None:
        """A caller may normalise in place; the payload must stay pristine."""
        payload = {0: _panel_rad_record(0.4)}
        path = _dump(tmp_path, payload, "in_place")
        materialiser = Materialiser(_config(), cache_size=1)
        first = materialiser.materialise(_ref(path))
        first.phase_cos[:] = 99.0
        second = materialiser.materialise(_ref(path))
        assert second.phase_cos.max() < 10.0

    def test_payload_is_not_mutated_on_disk(self, tmp_path: Path) -> None:
        """Materialising must not write anything back into the payload."""
        payload = {0: _panel_rad_record(0.45)}
        path = _dump(tmp_path, payload, "pristine")
        before = pickle.loads(path.read_bytes())
        materialiser = Materialiser(_config())
        materialiser.materialise(_ref(path))
        materialiser.materialise(_ref(path))
        after = pickle.loads(path.read_bytes())
        np.testing.assert_array_equal(before[0]["_phase"], after[0]["_phase"])
        np.testing.assert_array_equal(before[0]["_img"], after[0]["_img"])

    def test_phase_grid_output_owns_its_data(self, tmp_path: Path) -> None:
        """The standalone phase entry point must be safe to mutate too."""
        path = _dump(tmp_path, {0: _panel_rad_record()}, "phase_own")
        cos_grid, sin_grid = Materialiser(_config()).phase_grid(_ref(path))
        assert cos_grid.flags.owndata
        assert sin_grid.flags.owndata
        assert not np.shares_memory(cos_grid, sin_grid)


class TestHwSampleShape:
    """The sample container itself."""

    def test_fields_and_order(self, tmp_path: Path) -> None:
        """``HwSample`` must carry the five documented fields."""
        path = _dump(tmp_path, {0: _panel_rad_record()}, "fields")
        sample = Materialiser(_config()).materialise(_ref(path))
        assert isinstance(sample, HwSample)
        for name in ("phase_cos", "phase_sin", "image", "exposure_ms", "contrast"):
            assert hasattr(sample, name)
        assert sample.phase_cos.shape == (GRID, GRID)
        assert sample.image.shape == (GRID, GRID)

    def test_is_frozen(self, tmp_path: Path) -> None:
        """A sample must not be reassigned after the fact."""
        path = _dump(tmp_path, {0: _panel_rad_record()}, "frozen")
        sample = Materialiser(_config()).materialise(_ref(path))
        with pytest.raises(AttributeError):
            sample.image = np.zeros((GRID, GRID), dtype=np.float32)  # type: ignore[misc]

    def test_equality_is_identity_and_never_raises(self, tmp_path: Path) -> None:
        """``==`` must return a plain ``bool``, never ``bool()`` of an array.

        This is why ``eq=False`` is set on ``HwSample``: the generated
        ``__eq__`` would compare the numpy fields element-wise and then ``bool()``
        the result, raising ``ValueError: The truth value of an array with more
        than one element is ambiguous``. Identity equality is the correct semantic
        for an owning array container; value equality is ``np.array_equal`` on the
        fields. Two independently materialised samples are therefore ``!=`` even
        though they carry equal numbers.
        """
        path = _dump(tmp_path, {0: _panel_rad_record()}, "eq")
        materialiser = Materialiser(_config())
        first = materialiser.materialise(_ref(path))
        second = materialiser.materialise(_ref(path))
        assert isinstance(first == second, bool)
        assert (first == second) is False
        assert (first == first) is True
        assert isinstance(first == 3, bool)
        np.testing.assert_array_equal(first.phase_cos, second.phase_cos)
