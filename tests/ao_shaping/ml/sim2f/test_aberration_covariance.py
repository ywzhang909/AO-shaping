"""The correlated (von Karman) aberration option must not disturb the i.i.d. one.

Two separate things are pinned here.

**Backward compatibility.** ``draw_aberration`` consumed the RNG in a fixed
order, and every corpus so far was generated with it.  Adding
``aberration_model`` must leave that byte-identical, so ``_PINNED_COEFFS`` below
was recorded by running the *pre-change* code
(``git stash``-equivalent: the numbers were captured before the new branch
existed) for the default ``Sim2fConfig()`` bench.  It is **not** a
self-comparison: a regression that changes the draw order fails these.

**The new ensemble.** ``aberration_model="kolmogorov"`` draws
``N(0, Sigma)`` from the packaged covariance asset and *then* applies the same
``rms_rad`` rescale.  Two consequences are tested rather than assumed:

* the drawn vector's RMS is still exactly the uniform ``rms_rad`` draw -- the
  amplitude knob, i.e. the difficulty of a state, is untouched;
* the *shape* agrees with ``Sigma`` once the per-draw rescale is factored out.
  That rescale puts every draw on a sphere of radius ``rms * sqrt(K)``, so the
  absolute covariance of the rescaled draws is ``Sigma`` times a scalar that
  depends on the ``rms_rad`` distribution.  The invariant is therefore the
  **correlation matrix** and the **normalised diagonal**, not ``Sigma`` itself;
  comparing the raw covariance would have made the test depend on the RMS knob.

Tolerances are stated as the measured value plus margin, never as a round number
picked to pass (see the docstrings for the measurements).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.sim2f.config import AberrationConfig, Sim2fConfig
from ml.sim2f.optics import (
    DEFAULT_COVARIANCE_PATH,
    Sim2f,
    _covariance_factor,
    draw_aberration,
)

#: ``draw_aberration(Sim2f(Sim2fConfig(), device="cpu"), AberrationConfig(),
#: default_rng(seed)).coeffs`` recorded from the code **before**
#: ``aberration_model`` existed.  Values are float64 radians for Noll 2..22.
_PINNED_COEFFS: dict[int, tuple[float, ...]] = {
    0: (
        0.016933865862953087,
        0.016782031813139676,
        0.03201503814090328,
        -0.033638231127647966,
        0.163072612152731,
        0.026711010466967677,
        -0.1363989919331166,
        0.09207396100807043,
        0.3320411816469369,
        0.2411578767095587,
        -0.1791940729884099,
        -0.3222178113388055,
        -0.1587061210650559,
        0.010522949800823241,
        -0.5920290943697986,
        -0.055711533828522106,
        -0.317249792051561,
        -0.18645928629009015,
        -0.13858618829408548,
        -0.08054039420331745,
        0.10481463571267685,
    ),
    1: (
        -0.11996133359106291,
        0.6202011322660562,
        0.09067333037877894,
        0.21557367229607938,
        0.08669907613296478,
        -0.3419184352466146,
        0.23754452169452175,
        0.11711840414581384,
        -0.14088415855158692,
        0.15247200265748137,
        0.09565539768718934,
        0.07717359085505227,
        0.007457341325524205,
        0.14344489243358136,
        -0.19322858589256778,
        -0.04274381721023445,
        -0.12649700051218274,
        0.1571234498889367,
        0.01042216587948161,
        -0.07673391379195513,
        -0.2051547668012367,
    ),
    7: (
        0.096918989352389,
        0.15317055941975996,
        0.0006916233701635627,
        0.1679623066637526,
        -0.15412724459252147,
        -0.5007132852297203,
        -0.2556274295742305,
        -0.5575288058240783,
        0.033814256465830646,
        0.7535029508612658,
        -0.27673096944074876,
        -0.34884670167755494,
        0.2754016054471606,
        0.20065091384734007,
        0.05926656031276192,
        -0.5231326980463126,
        -0.016446115366361553,
        0.39091706388611314,
        -0.7557514595730929,
        -0.2572831695865063,
        -1.0689155711638894,
    ),
    12345: (
        0.03684428917402089,
        0.05797321899192856,
        -0.1833653965084954,
        0.16274757352273703,
        -0.11212700347170117,
        -0.033377277243607124,
        -0.009702986680615018,
        -0.0954138356318088,
        -0.17614934748469518,
        0.08356678869891944,
        0.046498384540121514,
        -0.25149684877551676,
        0.3023080020525848,
        0.12472657425848759,
        -0.09779665891304538,
        0.11618839397018926,
        -0.060135937762428274,
        -0.007815818223589363,
        0.1015902602586778,
        -0.16183831904411794,
        0.07416103717527982,
    ),
}


@pytest.fixture(scope="module")
def bench() -> Sim2f:
    """The reference bench (default ``Sim2fConfig``): one build per module.

    The geometry matters -- ``_PINNED_COEFFS`` was recorded on this exact bench,
    and tip/tilt goes through the bench's own tilt scale.
    """
    return Sim2f(Sim2fConfig(), device="cpu")


def _kolmogorov(**overrides) -> AberrationConfig:
    """``AberrationConfig`` on the packaged asset, with overrides."""
    payload = {"aberration_model": "kolmogorov", "covariance_path": str(DEFAULT_COVARIANCE_PATH)}
    payload.update(overrides)
    return AberrationConfig(**payload)


def _draw_many(bench: Sim2f, cfg: AberrationConfig, draws: int, seed: int) -> np.ndarray:
    """``(draws, n_modes)`` coefficient matrix from consecutive seeds."""
    return np.asarray(
        [draw_aberration(bench, cfg, np.random.default_rng(seed + i)).coeffs
         for i in range(draws)],
        dtype=np.float64,
    )


def _correlation(matrix: np.ndarray) -> np.ndarray:
    """Correlation matrix of a covariance matrix."""
    scale = np.sqrt(np.diag(matrix))
    return matrix / np.outer(scale, scale)


#: Noll-index blocks of one radial order ``n``: order ``n`` starts at
#: ``n * (n + 1) / 2 + 1``, so for ``n = 2..6`` the blocks are these.  Within a
#: block the modes are symmetry partners and their variances are equal in exact
#: arithmetic, so anything that compares "how much energy sits at high order" has
#: to average over a block rather than compare neighbours.
_ORDER_BLOCKS: tuple[tuple[int, int], ...] = (
    (4, 6),
    (7, 10),
    (11, 15),
    (16, 21),
    (22, 22),
)


def _order_means(values: np.ndarray, noll: np.ndarray) -> list[float]:
    """Mean of ``values`` (indexed like ``noll``) over each radial order."""
    return [
        float(values[(noll >= low) & (noll <= high)].mean())
        for low, high in _ORDER_BLOCKS
    ]


class TestIidIsBitIdentical:
    """``aberration_model="iid"`` (the default) must not move by one ulp."""

    @pytest.mark.parametrize("seed", sorted(_PINNED_COEFFS))
    def test_default_draw_reproduces_pre_change_coefficients(self, bench, seed):
        """The pinned values come from the pre-change code, not from a rerun."""
        state = draw_aberration(bench, AberrationConfig(), np.random.default_rng(seed))
        assert np.array_equal(np.asarray(state.coeffs), np.asarray(_PINNED_COEFFS[seed]))

    def test_default_model_is_iid(self):
        assert AberrationConfig().aberration_model == "iid"
        assert AberrationConfig().covariance_path is None

    def test_naming_the_model_explicitly_changes_nothing(self, bench):
        """``aberration_model="iid"`` + a path is the same draw as the default."""
        default = draw_aberration(bench, AberrationConfig(), np.random.default_rng(11))
        explicit = draw_aberration(
            bench,
            AberrationConfig(aberration_model="iid", covariance_path="does/not/exist.npz"),
            np.random.default_rng(11),
        )
        assert np.array_equal(np.asarray(default.coeffs), np.asarray(explicit.coeffs))
        assert default.illum_center_px == explicit.illum_center_px


class TestConfigContract:
    """The dataclass refuses a broken contract where it is cheap to notice."""

    def test_kolmogorov_without_a_path_is_rejected(self):
        with pytest.raises(ValueError, match="covariance_path"):
            AberrationConfig(aberration_model="kolmogorov")

    def test_missing_file_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="not a readable file"):
            AberrationConfig(
                aberration_model="kolmogorov", covariance_path="no/such/covariance.npz"
            )

    def test_iid_ignores_the_path(self):
        """A stale path under "iid" is not an error: only that model reads it."""
        cfg = AberrationConfig(covariance_path="no/such/covariance.npz")
        assert cfg.covariance_path == "no/such/covariance.npz"

    def test_unknown_model_is_rejected(self):
        with pytest.raises(ValueError, match="aberration_model"):
            AberrationConfig(aberration_model="turbulent")

    def test_to_dict_round_trips_the_new_fields(self):
        payload = _kolmogorov().to_dict()
        rebuilt = AberrationConfig(**payload)
        assert rebuilt.aberration_model == "kolmogorov"
        assert rebuilt.covariance_path == str(DEFAULT_COVARIANCE_PATH)
        assert rebuilt.n_modes == 21


class TestCovarianceAsset:
    """The packaged matrix, and the factor derived from it."""

    def test_asset_is_symmetric_positive_definite(self):
        with np.load(DEFAULT_COVARIANCE_PATH) as data:
            covariance = np.asarray(data["covariance"], dtype=np.float64)
            noll = np.asarray(data["noll"])
        assert covariance.shape == (19, 19)
        assert list(noll) == list(range(4, 23))
        assert np.array_equal(covariance, covariance.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        assert eigenvalues.min() > 0.0
        # Measured: min/max = 1.3e-2, so this is a well-conditioned matrix and
        # the clipping in `_covariance_factor` removes nothing.
        assert eigenvalues.min() / eigenvalues.max() > 1e-3

    def test_factor_reproduces_the_covariance(self):
        noll, factor = _covariance_factor(str(DEFAULT_COVARIANCE_PATH))
        with np.load(DEFAULT_COVARIANCE_PATH) as data:
            covariance = np.asarray(data["covariance"], dtype=np.float64)
        assert noll.shape == (factor.shape[0],)
        worst = np.abs(factor @ factor.T - covariance).max() / np.abs(covariance).max()
        assert worst < 1e-12

    def test_clipping_keeps_a_near_singular_matrix_finite(self, tmp_path):
        """A round-off negative eigenvalue must not reach ``sqrt``.

        ``Sigma`` from a pinv projection can carry eigenvalues of order
        ``-1e-16 * max``; ``sqrt`` of that is NaN and every draw becomes NaN.
        """
        rng = np.random.default_rng(0)
        white = rng.standard_normal((17, 17))
        covariance = white @ white.T
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues[0] = -1e-15 * abs(eigenvalues[-1])
        broken = (eigenvectors * eigenvalues) @ eigenvectors.T
        path = tmp_path / "near_singular.npz"
        np.savez(path, covariance=broken, noll=np.arange(4, 21))

        _, factor = _covariance_factor(str(path))
        assert np.all(np.isfinite(factor))
        assert np.allclose(factor @ factor.T, broken, atol=1e-12 * abs(broken).max())

    def test_asset_with_the_wrong_mode_count_is_a_clear_error(self, bench):
        cfg = _kolmogorov(n_modes=15)
        with pytest.raises(ValueError, match="covers 19 modes"):
            draw_aberration(bench, cfg, np.random.default_rng(0))

    def test_vanished_asset_raises_filenotfound(self, tmp_path):
        """A path that dies after construction surfaces at the load, not later."""
        asset = tmp_path / "gone.npz"
        np.savez(asset, covariance=np.eye(19), noll=np.arange(4, 23))
        cfg = _kolmogorov(covariance_path=str(asset))
        asset.unlink()
        with pytest.raises(FileNotFoundError, match="build_sim2f_aberration_covariance"):
            draw_aberration(bench, cfg, np.random.default_rng(0))


#: Ladder-compression bounds for
#: :meth:`TestKolmogorovDraw.test_end_to_end_shape_survives_the_rms_rescale`, measured
#: on the reference bench with the packaged asset at the test's own budget (2000
#: end-to-end draws, TODO.md 8.9 item 103).  The two bands below bracket the **two
#: endpoints** the docstring quotes -- the share ratio at the lowest probed mode
#: (Noll 4) and at the highest (Noll 22).  Measured at seed 20261025: 0.8538 and
#: 1.2739; over seeds 1 / 8 / 21 they stay in 0.893..0.897 and 1.315..1.319.
_LADDER_LOW_BAND = (0.80, 0.95)
_LADDER_HIGH_BAND = (1.15, 1.35)

#: ``max`` over all 21 diagonal share ratios is a **different statistic** and it is
#: positively biased: the maximum of noisy entries sits about a sampling sigma above
#: the typical one.  Measured at N=2000 it is 1.3466 (seed 20261025, on Noll 17) and
#: 1.3559..1.3660 on seeds 1 / 8 / 21, while at N=8000 it falls to 1.2860..1.2988 --
#: the shrink with N is the evidence that the gap is sampling bias, not drift.  The
#: bound is therefore pinned above the N=2000 measurements with ~1.06x margin, never
#: at the endpoint value: the old ``1.30`` was the endpoint number, so it lay below
#: this statistic's expected value and no seed at this budget could satisfy it.
_LADDER_MAX_OVER_ENTRIES = 1.45

#: Floor for the weakest entry, kept from the original assertion.  The minimum over
#: 21 entries is biased the other way; measured 0.8538 at seed 20261025.
_LADDER_MIN_OVER_ENTRIES = 0.80

class TestKolmogorovDraw:
    """The correlated ensemble keeps the amplitude knob and changes only shape."""

    def test_drawn_rms_is_still_the_uniform_draw(self, bench):
        """The rescale happens *after* the correlated draw, unchanged.

        Replays the RNG to recover the exact ``rms_rad`` sample the call made
        (one ``standard_normal(n_modes - 2)`` then one ``uniform``), so this is
        an equality, not a range check.
        """
        cfg = _kolmogorov()
        for seed in (0, 3, 17):
            rng = np.random.default_rng(seed)
            rng.standard_normal(cfg.n_modes - 2)
            expected = float(rng.uniform(*cfg.rms_rad))
            state = draw_aberration(bench, cfg, np.random.default_rng(seed))
            measured = float(np.sqrt(np.mean(np.asarray(state.coeffs[2:]) ** 2)))
            assert measured == pytest.approx(expected, rel=1e-12)
            assert cfg.rms_rad[0] <= measured <= cfg.rms_rad[1]

    def test_amplitude_range_is_identical_to_iid(self, bench):
        """Same seeds, same RMS draws: only the direction moved."""
        correlated = _draw_many(bench, _kolmogorov(), 64, seed=100)
        independent = _draw_many(bench, AberrationConfig(), 64, seed=100)
        rms_correlated = np.sqrt(np.mean(correlated[:, 2:] ** 2, axis=1))
        rms_independent = np.sqrt(np.mean(independent[:, 2:] ** 2, axis=1))
        # The rescale is `high * (target / |high|)`, so both land on the same
        # `target` up to the last ulp of the division.
        assert np.allclose(rms_correlated, rms_independent, rtol=1e-15, atol=0.0)

    def test_tilt_and_illumination_are_unchanged_by_the_model(self, bench):
        """The rest of the state is drawn after the branch, from the same stream."""
        for seed in (0, 5):
            correlated = draw_aberration(bench, _kolmogorov(), np.random.default_rng(seed))
            independent = draw_aberration(bench, AberrationConfig(), np.random.default_rng(seed))
            assert np.array_equal(correlated.coeffs[:2], independent.coeffs[:2])
            assert correlated.illum_center_px == independent.illum_center_px
            assert correlated.illum_width_scale == independent.illum_width_scale

    def test_empirical_covariance_matches_the_target(self, bench):
        """2000 raw correlated draws reproduce ``Sigma`` within sampling error.

        This is the covariance check proper: it exercises exactly the draw
        mechanism (``L @ g`` with ``L L^T = Sigma``) *before* the RMS rescale, so
        the target is the asset itself rather than the asset times an unknown
        scalar.

        Measured with 2000 draws: ``max |emp - Sigma| / max |Sigma| = 0.0196``
        and the mean relative error of a diagonal entry is 1.7%.  The per-entry
        sampling error of a covariance is ``sqrt(2/N) = 3.2%``, so the stated
        0.04 (2x the measurement) is ~1.3 sampling sigma at the worst entry of
        361 -- a passing margin, not a rubber stamp.
        """
        noll, factor = _covariance_factor(str(DEFAULT_COVARIANCE_PATH))
        generator = np.random.default_rng(20261025)
        draws = np.stack(
            [factor @ generator.standard_normal(noll.size) for _ in range(2000)]
        )
        empirical = np.cov(draws, rowvar=False)
        with np.load(DEFAULT_COVARIANCE_PATH) as data:
            target = np.asarray(data["covariance"], dtype=np.float64)
        assert np.abs(empirical - target).max() / np.abs(target).max() < 0.04

    def test_end_to_end_shape_survives_the_rms_rescale(self, bench):
        """Through ``draw_aberration``, the target structure is preserved.

        The rescale is not a no-op on the shape: forcing every draw onto a
        sphere of radius ``rms * sqrt(K)`` removes variance along the dominant
        mode, which is the Bingham shrinkage of an anisotropic shape under a
        fixed norm.  Measured on 2000 end-to-end draws (seed 20261025):

        * ``max |corr - corr(Sigma)|`` off-diagonal: **0.091** (rms 0.029);
        * the diagonal ladder is **compressed**: the lowest probed mode (Noll 4)
          keeps 0.8538 of its target share, the highest (Noll 22) gains 1.2739x, so
          the defocus-to-Noll-22 ratio falls 35x -> 24.6x (an i.i.d. draw would be
          1x, which is the whole point of the change).

        The endpoints above are the statistics quoted in ``draw_aberration`` and they
        reproduce here exactly at the pinned seed; the bands in
        :data:`_LADDER_LOW_BAND` / :data:`_LADDER_HIGH_BAND` carry them with margin
        over the extra seeds listed there.  ``max`` over all 21 ratios is *not* one
        of those endpoints -- it lands on a mid-order mode (Noll 17 here) and is
        inflated by sampling, so it gets its own bound in
        :data:`_LADDER_MAX_OVER_ENTRIES`.  An earlier version of this test bounded
        that max with the endpoint number (``1.30``), which no seed at this draw
        count could satisfy -- it was red on arrival at the commit that added it
        (38303f58), with a byte-identical failure signature.  The bounds are still
        not tightened to the unconstrained sampling error (0.019): that would assert
        a property the amplitude knob deliberately breaks.  See
        ``draw_aberration``'s docstring for what to change if the *exact* ``Sigma``
        is required.
        """
        draws = _draw_many(bench, _kolmogorov(), 2000, seed=20261025)[:, 2:]
        empirical = np.cov(draws, rowvar=False)

        with np.load(DEFAULT_COVARIANCE_PATH) as data:
            target = np.asarray(data["covariance"], dtype=np.float64)
            noll = np.asarray(data["noll"])

        residual = _correlation(empirical) - _correlation(target)
        off_diagonal = ~np.eye(target.shape[0], dtype=bool)
        assert np.abs(residual[off_diagonal]).max() < 0.12
        assert np.sqrt(np.mean(residual[off_diagonal] ** 2)) < 0.05

        share = np.diag(empirical) / np.diag(empirical).mean()
        expected = np.diag(target) / np.diag(target).mean()
        ratio = share / expected
        low = int(np.flatnonzero(noll == 4)[0])
        high = int(np.flatnonzero(noll == 22)[0])
        assert _LADDER_LOW_BAND[0] < ratio[low] < _LADDER_LOW_BAND[1], ratio[low]
        assert _LADDER_HIGH_BAND[0] < ratio[high] < _LADDER_HIGH_BAND[1], ratio[high]
        assert ratio.max() < _LADDER_MAX_OVER_ENTRIES, (
            f"max over {ratio.size} entries landed on Noll "
            f"{int(noll[int(ratio.argmax())])} at {ratio.max():.4f}"
        )
        # Separate bounds only make sense because the two statistics are attained at
        # different modes: the max is a mid-order entry inflated by sampling, not the
        # highest mode.  If they ever merge, both bands need re-measuring.
        assert int(noll[int(ratio.argmax())]) != int(noll[high]), (
            "max over entries now coincides with the highest mode: the endpoint and "
            "max-over-entries statistics have merged, re-measure both bounds"
        )
        assert ratio.min() > _LADDER_MIN_OVER_ENTRIES, ratio.min()
        # The ladder survives: strictly decreasing *between radial orders* (modes
        # within one order are symmetry partners and only differ by sampling
        # noise), and low order still dominates by more than 15x.
        by_order = _order_means(share, noll)
        assert all(later < earlier for earlier, later in zip(by_order, by_order[1:]))
        assert by_order[0] / by_order[-1] > 15.0

    def test_energy_is_low_order_dominated(self, bench):
        """The physical point of the change: i.i.d. spends 95% off low order.

        Grouped by radial order of the Zernike index, the target gives
        ``n=2 ~61%, n=3 ~22%, n=4 ~11%, n=5 ~6%``, while an i.i.d. draw would
        split it evenly at ``1/19 = 5%`` per mode (i.e. 16% / 21% / 26% / 32%).
        """
        with np.load(DEFAULT_COVARIANCE_PATH) as data:
            covariance = np.asarray(data["covariance"], dtype=np.float64)
            noll = np.asarray(data["noll"])
        target = np.diag(covariance)
        shares = _order_means(target / target.mean(), noll)
        assert shares[0] > 3.0  # defocus/astigmatism dominate
        assert all(later < earlier for earlier, later in zip(shares, shares[1:]))
        assert target[noll <= 6].sum() / target.sum() > 0.55


class TestRegenerationContract:
    """The asset must be traceable and re-checkable, not an opaque blob."""

    def test_provenance_records_the_parameters_it_depends_on(self):
        import json

        provenance = DEFAULT_COVARIANCE_PATH.with_suffix(".json")
        record = json.loads(provenance.read_text(encoding="utf-8"))
        assert record["asset"] == DEFAULT_COVARIANCE_PATH.name
        assert record["turbulence"]["r0_m_at_500nm"] > 0.0
        assert record["turbulence"]["L0_m"] > 0.0
        assert record["turbulence"]["wavelength_rad2_expressed_at_m"] > 0.0
        assert record["oopao"]["commit"]
        assert record["environment"]["numpy"]
        assert record["sim2f"]["noll"] == list(range(4, 23))
        assert record["sim2f"]["grid"] == 128
        assert record["sim2f"]["pupil_radius_px"] == 48.0

    def test_recorded_sha_matches_the_payload(self):
        import hashlib
        import json

        record = json.loads(
            DEFAULT_COVARIANCE_PATH.with_suffix(".json").read_text(encoding="utf-8")
        )
        with np.load(DEFAULT_COVARIANCE_PATH) as data:
            payload = np.ascontiguousarray(data["covariance"], dtype="<f8")
        assert hashlib.sha256(payload.tobytes()).hexdigest() == record["covariance_sha256"]
