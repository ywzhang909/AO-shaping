"""Run slm-gsnet — square far-field shaping via FREEFORM per-pixel SLM phase.

This is the Gerchberg-Saxton / differentiable "shaping" family runner: instead of
driving the SLM with a low-order Zernike basis (which physically cannot synthesise
a true square far-field — low-order Zernike modes are smooth and circularly
symmetric), it optimises the SLM's *free* per-pixel phase (a ``phase_grid²``
freeform DOF) with SPGD (gradient) or a black-box heuristic search. The target is
a uniform, high-energy square, scored by the combined quality score
(uniformity CV + encircled energy + aspect) — the same ``optimize_slm_square``
objective as the ``spgd-square`` command, but freeform-first.

Usage:
    python src/ao_shaping/main.py slm-gsnet                    # SPGD (default, no options)
    python src/ao_shaping/main.py slm-gsnet spgd [OPTIONS]     # SPGD gradient search
    python src/ao_shaping/main.py slm-gsnet heuristic [OPTIONS]  # black-box heuristic
    python src/ao_shaping/main.py slm-gsnet train [OPTIONS]    # offline GSNet training

Offline dry-run (no hardware) — options live on the subcommand, not the group:
    python src/ao_shaping/main.py slm-gsnet spgd --cam-type sim --epochs 50
    python src/ao_shaping/main.py slm-gsnet heuristic --cam-type sim --epochs 50 --algorithm ga
    python src/ao_shaping/main.py slm-gsnet train --epochs 1 --max-samples 32

``train`` never opens a device: it reads the ``data/debug`` pickles the
optimizers already recorded and trains FourierGSNet on them in simulation.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import click
import numpy as np
from loguru import logger

from ao_shaping.optimizer.wfless.slm_square_shaping import (
    SlmSquareConfig,
    optimize_slm_square,
)
from ml.gsnet_debug.train import GsnetTrainParams, run_offline_training
from ao_shaping.runners.runner_common import (
    CameraParams,
    HeuristicParams,
    ObjectiveParamsSquare,
    RunParams,
    SlmParams,
    SpgdParams,
    config_payload,
    parse_center,
    patch_sim_square_shaping,
    resolve_spgd_delta,
    with_params,
)
from ao_shaping.utils.io.file import Recorder, save_recorder_debug_artifacts
from ao_shaping.utils.io.cli_helpers import setup_coredumpy
from ao_shaping.display import AutoDisplay, FrameInfo
from ao_shaping.display.frames import (
    Image2DFrame,
    Image2DWithBucketFrame,
    EpochCurveFrame,
)
from ao_shaping.utils.image.beam_metrics import zero_order_center

# --- debug artifact fields -------------------------------------------------
# The square-shaping recorder stores the metric keys below. Image-like entries
# (leading-underscore ``_img``/``_diff``) are the CCD frames and are rendered by
# the shared artifact writer; the scalar keys drive the JSON sidecar + summary.

_DEBUG_SCALAR_KEYS = (
    "J",
    "quality",
    "cv",
    "ee",
    "ar",
    "side",
    "lr",
    "delta",
    "_diff",
    "exp_t",
    "max_brt",
    "mean_b",
    "target_mean_b",
    "best_quality",
    "_epoch",
)

_IMG_KEYS = ("_img",)
_1D_KEYS = ("_c", "_grad")

# Artifact directory names and the summary-plot title are part of the run
# contract, so the default objective must keep them byte-for-byte identical.
# Mirrors ``ObjectiveParamsSquare.objective``'s default.
_DEFAULT_SQUARE_OBJECTIVE = "quality"


class _FreeformSquareDisplay:
    """Live pygame view for freeform square shaping (slm-gsnet).

    Composes three panels using :class:`AutoDisplay`:
    - CCD far-field image with target box overlay
    - SLM phase (freeform, raw radians mapped to [0,1] for visibility)
    - Metric curve (quality / CV / EE over epochs)

    Used as a context manager so the window is always torn down.
    """

    DEFAULT_FRAME_SIZE = (500, 400)
    DEFAULT_DISPLAY_SIZE = (1550, 500)

    def __init__(
        self,
        target_side: int,
        frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
        display_size: tuple[int, int] = DEFAULT_DISPLAY_SIZE,
        margin: int = 10,
    ) -> None:
        self.target_side = target_side
        self._roi_center: tuple[int, int] | None = None
        self._quality_curve: list[float] = []
        self._cv_curve: list[float] = []
        self._ee_curve: list[float] = []
        self._epoch = 0
        self._closed = False

        frames = [
            FrameInfo(
                "ccd",
                "CCD Far-field",
                "Image2DWithBucketFrame",
                {
                    "target_shape": "square",
                    "target_size": float(target_side),
                    "target_aspect_ratio": 1.0,
                },
            ),
            FrameInfo("phase", "SLM Phase (freeform)", "Image2DFrame", {}),
            FrameInfo(
                "curve",
                "Quality / CV / EE",
                "EpochCurveFrame",
                {"y_min": 0.0, "y_max": 1.2},
            ),
        ]
        self._display = AutoDisplay(
            frames,
            frame_size=frame_size,
            display_size=display_size,
            margin=margin,
            grid=(3, 1),
        )

    @property
    def closed(self) -> bool:
        return self._closed

    def init_window(self) -> None:
        self._display.init_window()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._display.close()

    def set_roi_center(self, center: tuple[int, int]) -> None:
        """Set the frozen ROI center (in window-local coordinates)."""
        self._roi_center = center

    def update(
        self,
        measured: np.ndarray,
        phase: np.ndarray,
        quality: float,
        cv: float,
        ee: float,
        epoch: int,
        total_epochs: int | None = None,
        info: str = "",
    ) -> bool:
        """Render one frame; returns ``False`` once the window has been closed."""
        if self.closed:
            return False

        self._epoch = epoch
        self._quality_curve.append(quality)
        self._cv_curve.append(cv)
        self._ee_curve.append(ee)

        # Map raw signed radians to [0, 1] for display visibility
        phase_disp = self._phase_for_display(phase)

        frame_data = {
            "ccd": {
                "img": measured,
                "center": self._roi_center
                or (measured.shape[1] // 2, measured.shape[0] // 2),
                "r": self.target_side / 2.0,
                "target_shape": "square",
                "target_size": float(self.target_side),
                "target_aspect_ratio": 1.0,
            },
            "phase": {"img": phase_disp},
            "curve": {
                "value": quality,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "label": info,
            },
        }
        try:
            alive = self._display.render(frame_data, info=info)
        except Exception:
            alive = False
        if not alive:
            self._closed = True
        return bool(alive)

    @staticmethod
    def _phase_for_display(phase: np.ndarray) -> np.ndarray:
        """Map signed radian phase onto ``[0, 1]`` so its structure is always visible."""
        arr = np.asarray(phase, dtype=np.float64)
        if arr.size == 0:
            return arr
        arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        lo = float(arr.min())
        span = float(arr.max()) - lo
        if span <= 0.0:
            return np.zeros_like(arr)
        return (arr - lo) / span


def _save_debug_artifacts(
    res: Recorder,
    objective: "ObjectiveParamsSquare",
    root_dir: str,
) -> Any:
    """Write PNG/pkl/json debug artifacts for the recorded square-shaping search.

    Delegates to the shared :func:`ao_shaping.utils.io.file.save_recorder_debug_artifacts`
    with the square-shaping key set. The JSON sidecar carries the objective
    config so the run is reproducible from the artifact directory alone.

    A non-default objective (``pearson``) is appended to the subdirectory and
    title so those runs are distinguishable; the default objective keeps its
    historical ``slm_gsnet_<name>`` path unchanged.
    """
    is_default = objective.objective == _DEFAULT_SQUARE_OBJECTIVE
    prefix = f"slm_gsnet_{objective.name}"
    title = f"slm-gsnet {objective.name} search"
    if not is_default:
        prefix = f"{prefix}_{objective.objective}"
        title = f"slm-gsnet {objective.name} ({objective.objective}) search"

    return save_recorder_debug_artifacts(
        res,
        root_dir=root_dir,
        subdir_prefix=prefix,
        scalar_keys=_DEBUG_SCALAR_KEYS,
        img_keys=_IMG_KEYS,
        d1_keys=_1D_KEYS,
        json_payload=_config_payload(objective),
        title=title,
    )


def _config_payload(obj: Any) -> dict[str, Any]:
    """Serialize a parameter dataclass for the JSON debug sidecar."""
    return config_payload(obj)


# --- parameter dataclasses -------------------------------------------------


@dataclass
class SlmGsnetConfig:
    """Aggregates every parameter group for one slm-gsnet run."""

    run: RunParams
    camera: CameraParams
    slm: SlmParams
    objective: ObjectiveParamsSquare
    search: SpgdParams | HeuristicParams


# --- shared execution helpers ------------------------------------------------


def _build_square_config(cfg: SlmGsnetConfig) -> SlmSquareConfig:
    """Build the optimizer config object directly (no flat kwargs flattening).

    Freeform per-pixel phase is the ONLY DOF that can synthesise a square
    far-field (low-order Zernike is smooth and cannot). This runner is the
    freeform-first entry point of the square-shaping family.
    ``center`` / ``epochs`` are not part of the config — they stay positional
    arguments of :func:`optimize_slm_square`.
    """
    cam = cfg.camera
    slm = cfg.slm
    obj = cfg.objective
    search = cfg.search

    zernike_radius: float | int | None = (
        slm.zernike_radius if slm.zernike_radius > 0 else None
    )

    common: dict[str, Any] = dict(
        n_max=slm.n_max,
        target_side=obj.target_side,
        target_mean_brightness=obj.target_mean_brightness,
        side_factor=obj.side_factor,
        exposure_time_ms=cam.exposure_time_ms,
        cam_id=cam.cam_id,
        cam_type=cam.cam_type,
        show=search.show,
        cam_size=cam.cam_size,
        target_max_brightness=obj.target_max_brightness,
        slm_number=slm.slm_number,
        slm_wavelength=slm.slm_wavelength,
        w_uniformity=obj.w_uniformity,
        w_efficiency=obj.w_efficiency,
        w_aspect=obj.w_aspect,
        # Background suppression (PBR). Off by default (0.0); the ROI energy
        # guard already blocks the light-scattering failure mode that PBR also
        # discourages, so this stays opt-in on top of a safe default.
        w_pbr=obj.w_pbr,
        objective=obj.objective,
        basis="freeform",
        phase_grid=24,
        zernike_radius=zernike_radius,
        random_seed=cfg.run.seed,
        # In-ROI energy guard, armed from the initial flat frame; 0 disables it.
        # HARDWARE: without it the 2026-10-01 run traded 6x of encircled energy
        # (0.158 -> 0.026) for a +0.0118 uniformity gain and ended 35.9% worse,
        # with dec=0.487 (random walk). Guarded epochs are SKIPPED, not merely
        # penalised. See report/fouriergsnet_pipeline/hardware_run_20261001.md.
        max_roi_energy_loss=0.6,
        # Start from flat (the measured best-focus state, FWHM 13.6px /
        # hollowness 0.92) rather than the previous uniform(-pi, pi), which
        # destroyed the focus (0-order peak 225 -> 17).
        init_amplitude_rad=0.0,
    )

    if isinstance(search, SpgdParams):
        delta, delta_pinned = resolve_spgd_delta(search.delta)
        common.update(
            algorithm="spgd",
            pop_size=None,
            lr=search.lr,
            delta=delta,
            # An explicit --delta must survive the lr==0 adaptive schedule.
            delta_pinned=delta_pinned,
            optimizer_type=search.optimizer_type,
        )
    else:  # HeuristicParams — black-box search has no learning rate / perturbation
        common.update(
            algorithm=search.algorithm,
            pop_size=search.pop_size,
            lr=0.0,
        )

    return SlmSquareConfig(**common)


def _maybe_sim_patch(cam_type: str) -> None:
    """Wire the pure-numpy 2f-Fourier sim into the square-shaping optimizer.

    Thin alias kept for back-compat: the implementation now lives in
    ``runner_common.patch_sim_square_shaping`` so ``spgd-square``
    (``runners/slm/slm_shaping_runner.py``) and this runner cannot drift into
    patching different module sets. The module-level name is what
    ``_execute`` resolves at call time, so monkeypatching
    ``gsnet_runner._maybe_sim_patch`` still works.
    """
    patch_sim_square_shaping(cam_type)


# --- click group + subcommands ---------------------------------------------


@click.group(invoke_without_command=True)
@click.pass_context
@with_params(RunParams, kw_name="run")
def run(ctx: click.Context, run: RunParams) -> None:
    """Square far-field shaping via FREEFORM per-pixel SLM phase.

    The phase basis is always freeform (per-pixel) — the only DOF that can
    synthesise a true square far-field (low-order Zernike cannot). Without a
    subcommand, the SPGD (gradient) search runs.

    Subcommands: ``spgd`` (gradient search), ``heuristic`` (black-box search)
    and ``train`` (offline FourierGSNet training on the recorded debug corpus —
    the only one that touches no hardware).
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(spgd, **ctx.params)


@click.command(name="spgd")
@click.pass_context
@with_params(RunParams, kw_name="run")
@with_params(CameraParams, kw_name="camera")
@with_params(SlmParams, kw_name="slm")
@with_params(ObjectiveParamsSquare, kw_name="objective")
@with_params(SpgdParams, kw_name="search")
def spgd(
    ctx: click.Context,
    run: RunParams,
    camera: CameraParams,
    slm: SlmParams,
    objective: ObjectiveParamsSquare,
    search: SpgdParams,
) -> None:
    """Run the SPGD (Stochastic Parallel Gradient Descent) search."""
    _execute(SlmGsnetConfig(run, camera, slm, objective, search))


@click.command(name="heuristic")
@click.pass_context
@with_params(RunParams, kw_name="run")
@with_params(CameraParams, kw_name="camera")
@with_params(SlmParams, kw_name="slm")
@with_params(ObjectiveParamsSquare, kw_name="objective")
@with_params(HeuristicParams, kw_name="search")
def heuristic(
    ctx: click.Context,
    run: RunParams,
    camera: CameraParams,
    slm: SlmParams,
    objective: ObjectiveParamsSquare,
    search: HeuristicParams,
) -> None:
    """Run a black-box heuristic freeform search (ga/pso/sa/hc/rs/cem/de)."""
    _execute(SlmGsnetConfig(run, camera, slm, objective, search))


@click.command(name="train")
@click.pass_context
@with_params(RunParams, kw_name="run")
@with_params(GsnetTrainParams, kw_name="params")
def train(ctx: click.Context, run: RunParams, params: GsnetTrainParams) -> None:
    """Train FourierGSNet offline on the recorded debug corpus (no hardware).

    Streams the ``data/debug`` pickles produced by the optimizers, trains the
    unrolled GS network, evaluates it in simulation and writes
    ``summary.json``, ``comparison.png``, ``train_history.png`` and the best
    checkpoint. Artifacts land in ``<dir>/gsnet_train/run-<timestamp>/`` unless
    ``--out-dir`` says otherwise.
    """
    run_offline_training(run, params)


run.add_command(spgd, name="spgd")
run.add_command(heuristic, name="heuristic")
run.add_command(train, name="train")


def _execute(cfg: SlmGsnetConfig) -> None:
    """Shared execution path for both search families.

    The optimizer (:func:`optimize_slm_square`) owns the live pygame display during
    the search when ``config.show`` is True. For freeform basis the internal
    ``_SPGDDisplay`` still renders the SLM phase and CCD image correctly; the
    Zernike coefficient panel is empty (freeform has 576 DOF, not Zernike modes).
    This function adds a post-run best-result viewer when ``--show`` is used.
    """
    setup_coredumpy()
    _maybe_sim_patch(cfg.camera.cam_type)

    # The optimizer creates its own live display when show=True. We pass the flag
    # through the config so the internal _SPGDDisplay is used during optimization.
    config = _build_square_config(cfg)
    res = optimize_slm_square(
        center=parse_center(cfg.camera.center),
        epochs=cfg.search.epochs,
        config=config,
    )

    if cfg.run.debug:
        _save_debug_artifacts(res, cfg.objective, cfg.run.dir)

    # Post-run best-result viewer (opens after optimization completes).
    # This gives a clean view of the best phase + far field without the
    # coefficient-panel clutter from the internal display.
    if cfg.search.show:
        _show_best_result(res, config)

    final = res.history[-1]
    logger.info(
        "SLM GSNET done. Final quality={:.4f} (CV={:.4f}, EE={:.4f}, AR={:.4f})",
        final.get("quality", float("nan")),
        final.get("cv", float("nan")),
        final.get("ee", float("nan")),
        final.get("ar", float("nan")),
    )


def _show_best_result(res: Recorder, config: SlmSquareConfig) -> None:
    """Display the best far-field frame and SLM phase after optimization.

    Opens a pygame window showing:
    - Best CCD far-field image with target box
    - Best SLM phase (freeform, mapped to [0,1] for visibility)
    - Metric summary text
    Press ESC or close window to exit.
    """
    best_iter, _ = res.get_best_iter()
    best_img = best_iter.get("_img")
    best_phase = best_iter.get("_c")
    if best_img is None or best_phase is None:
        logger.warning("No best frame/phase recorded; skipping result viewer")
        return

    # Reconstruct the full-panel phase for display (freeform: upsample grid -> panel)
    phase_grid = int(config.phase_grid)
    panel_h, panel_w = 1200, 1920
    cells = np.asarray(best_phase, dtype=np.float64).reshape(phase_grid, phase_grid)
    block_h = int(np.ceil(panel_h / phase_grid))
    block_w = int(np.ceil(panel_w / phase_grid))
    upsampled = np.kron(cells, np.ones((block_h, block_w), dtype=np.float64))
    best_phase_panel = upsampled[:panel_h, :panel_w]

    # Map phase to [0,1] for display
    arr = np.nan_to_num(best_phase_panel, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(arr.min()), float(arr.max())
    phase_disp = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

    # Locate target center on best frame (use zero-order as proxy)
    from ao_shaping.utils.image.beam_metrics import zero_order_center

    center = zero_order_center(best_img, refine=False)
    target_side = int(config.target_side) if config.target_side > 0 else 0

    import pygame

    try:
        pygame.init()
    except pygame.error as e:
        logger.warning("pygame 初始化失败 (无显示环境?), 跳过结果查看器: {}", e)
        return

    try:
        # Window layout: 2 panels side by side + info bar
        panel_w, panel_h = 600, 500
        info_h = 80
        win_w = panel_w * 2 + 20
        win_h = panel_h + info_h + 20
        screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("slm-gsnet Best Result")
        font = pygame.font.SysFont("consolas", 18)
        title_font = pygame.font.SysFont("consolas", 22, bold=True)
        clock = pygame.time.Clock()

        # Pre-render surfaces
        def to_surface(arr: np.ndarray, cmap: str = "gray") -> pygame.Surface:
            arr = np.asarray(arr, dtype=np.float64)
            if arr.size == 0:
                arr = np.zeros((16, 16), dtype=np.float64)
            vmin, vmax = float(arr.min()), float(arr.max())
            if vmax - vmin < 1e-9:
                norm = np.zeros_like(arr, dtype=np.uint8)
            else:
                norm = ((arr - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
            if cmap == "heat" and norm.ndim == 2:
                f = norm.astype(np.float64) / 255.0
                r = np.clip(1.5 - np.abs(4 * f - 3.0), 0.0, 1.0)
                g = np.clip(1.5 - np.abs(4 * f - 2.0), 0.0, 1.0)
                b = np.clip(1.5 - np.abs(4 * f - 1.0), 0.0, 1.0)
                rgb = np.dstack((r, g, b))
                rgb = (rgb * 255.0).astype(np.uint8)
                return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            if norm.ndim == 2:
                norm = np.dstack((norm, norm, norm))
            return pygame.surfarray.make_surface(norm.swapaxes(0, 1))

        img_surf = to_surface(best_img, "heat")
        phase_surf = to_surface(phase_disp, "gray")
        img_surf = pygame.transform.scale(img_surf, (panel_w, panel_h))
        phase_surf = pygame.transform.scale(phase_surf, (panel_w, panel_h))

        # Metrics text
        quality = best_iter.get("quality", float("nan"))
        cv = best_iter.get("cv", float("nan"))
        ee = best_iter.get("ee", float("nan"))
        ar = best_iter.get("ar", float("nan"))
        epoch = best_iter.get("_epoch", -1)
        lines = [
            f"Best @ epoch {epoch} | Quality={quality:.4f} | CV={cv:.4f} | EE={ee:.4f} | AR={ar:.4f}",
            f"Target side: {target_side or 'auto'} px | Basis: freeform ({phase_grid}x{phase_grid})",
            "Press ESC or close window to exit",
        ]

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            screen.fill((25, 25, 25))
            # Panels
            screen.blit(phase_surf, (10, 10))
            screen.blit(img_surf, (panel_w + 20, 10))

            # Panel titles
            screen.blit(
                title_font.render("SLM Phase (freeform)", True, (0, 255, 255)),
                (10, panel_h + 15),
            )
            screen.blit(
                title_font.render("CCD Far-field", True, (0, 255, 255)),
                (panel_w + 20, panel_h + 15),
            )

            # Metrics
            y = panel_h + 50
            for line in lines:
                screen.blit(font.render(line, True, (200, 200, 200)), (10, y))
                y += 24

            # Draw target box on CCD panel
            if target_side > 0:
                cx, cy = center
                scale_x = panel_w / best_img.shape[1]
                scale_y = panel_h / best_img.shape[0]
                box_x = int((cx - target_side / 2) * scale_x) + panel_w + 20
                box_y = int((cy - target_side / 2) * scale_y) + 10
                box_w = int(target_side * scale_x)
                box_h = int(target_side * scale_y)
                pygame.draw.rect(screen, (255, 0, 0), (box_x, box_y, box_w, box_h), 2)

            pygame.display.flip()
            clock.tick(30)
    finally:
        pygame.quit()


if __name__ == "__main__":
    run()
