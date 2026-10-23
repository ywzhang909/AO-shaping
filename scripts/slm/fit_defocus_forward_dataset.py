"""Fit a compact physical model to paired CCD data, keeping test phases untouched.

Uses canonical Zernike and Fraunhofer implementations. Camera mapping is fixed by
calibration ramps. Reports predictions on the original, fixed CCD coordinate grid.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def ramp_matrix(rows: list[dict]) -> np.ndarray:
    columns = []
    for axis in ("x", "y"):
        values = []
        for period in (480, 960):
            pair = {sign: next(r for r in rows if r["label"] == f"ramp_{axis}_{period}_{sign}")
                    for sign in (-1, 1)}
            point = lambda sign, _pair=pair: np.array([_pair[sign]["spot"][f"centroid_{k}"] for k in ("x", "y")])
            values.append((point(1) - point(-1)) * period / 2)
        columns.append(np.mean(values, axis=0))
    matrix = np.stack(columns, axis=1)
    if abs(np.linalg.det(matrix)) < 1:
        raise ValueError("ramp geometry is singular")
    return matrix


class PhysicalForwardFit:
    """Nine-parameter shared physical fit with fixed camera geometry."""

    def __init__(self, phases, images, camera_xy, matrix, focus_diopter, train_mask,
                 amplitude_center, *, learn_amplitude=False, initial_defocus=None,
                 learn_residual=False, fft_size=None, pixel_samples=1,
                 basis_arrays=None, batch_size=None):
        import torch
        from ao_shaping.utils.wavefront.fraunhofer import focal_intensity
        from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

        self.torch = torch
        self.propagate = focal_intensity
        self.phases = torch.tensor(phases, dtype=torch.float32)
        size = self.phases.shape[-1]
        if self.phases.ndim != 3 or self.phases.shape[-2] != size or size < 8:
            raise ValueError("phases must be square batched grids of size >= 8")
        self.pitch = 1920. / size
        self.fft_size = fft_size or 4 * size
        if self.fft_size < size or self.fft_size % 2 or pixel_samples < 1:
            raise ValueError("require even fft_size >= phase grid, pixel_samples >= 1")
        self.pixel_samples = pixel_samples
        self.batch_size = batch_size or len(phases)
        self.images = torch.tensor(images, dtype=torch.float32)
        self.xy = torch.tensor(camera_xy, dtype=torch.float32)
        self.inverse = torch.tensor(np.linalg.inv(matrix), dtype=torch.float32)
        self.train = torch.tensor(train_mask, dtype=torch.bool)
        self.z4 = torch.tensor(basis_arrays[4] if basis_arrays is not None else
                               np.nan_to_num(generate_zernike_phase({4: 1.0}, (size, size))),
                               dtype=torch.float32)
        # Z4 = sqrt(3)*(2*r^2 - 1); pupil half-span is 960 panel pixels.
        self.diopter_scale = float(np.pi * (960 * 8e-6) ** 2 / (2 * np.sqrt(3) * 1064e-9))
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        self.panel_x = x.float() * self.pitch
        self.panel_y = y.float() * self.pitch
        self.panel_mask = (self.panel_y < 1200).float()
        d = -focus_diopter if initial_defocus is None else initial_defocus
        self.d = torch.nn.Parameter(torch.tensor(float(d)))
        # Use one shared optical-axis intercept, not per-image fitted shifts.
        self.origin = torch.nn.Parameter(torch.tensor([1360., 1020.]))
        self.center = torch.nn.Parameter(torch.tensor(amplitude_center, dtype=torch.float32),
                                         requires_grad=learn_amplitude)
        self.log_width = torch.nn.Parameter(torch.log(torch.tensor([260., 260.])),
                                            requires_grad=learn_amplitude)
        params = [{"params": [self.d], "lr": .01}, {"params": [self.origin], "lr": 1.0}]
        if learn_amplitude:
            params += [{"params": [self.center], "lr": .5}, {"params": [self.log_width], "lr": .01}]
        self.residual_modes = list(range(5, 12)) if learn_residual else []
        self.residual = torch.nn.Parameter(torch.zeros(len(self.residual_modes)), requires_grad=learn_residual)
        self.residual_basis = torch.tensor(np.stack([
            basis_arrays[j] if basis_arrays is not None else
            np.nan_to_num(generate_zernike_phase({j: 1.}, (size, size)))
            for j in self.residual_modes
        ]), dtype=torch.float32) if self.residual_modes else None
        if learn_residual:
            params += [{"params": [self.residual], "lr": .03}]
        self.optimizer = torch.optim.Adam(params)
        self.history = []
        self.iteration_records = []

    def prediction(self):
        torch = self.torch
        width = self.log_width.exp()
        amplitude = torch.exp(-((self.panel_x - self.center[0]) / width[0]) ** 2
                              -((self.panel_y - self.center[1]) / width[1]) ** 2) * self.panel_mask
        phase = self.phases + self.d * self.diopter_scale * self.z4
        if self.residual_basis is not None:
            phase = phase + torch.einsum("i,ijk->jk", self.residual, self.residual_basis)
        grids = []
        for iy in range(self.pixel_samples):
            for ix in range(self.pixel_samples):
                offset = self.xy.new_tensor([(ix+.5)/self.pixel_samples-.5,
                                             (iy+.5)/self.pixel_samples-.5])
                frequencies = (self.xy + offset - self.origin) @ self.inverse.T
                fft_indices = frequencies * (self.fft_size * self.pitch) + self.fft_size//2
                grids.append(2 * fft_indices / (self.fft_size-1) - 1)
        predictions = []
        for start in range(0, len(phase), self.batch_size):
            field = amplitude * torch.exp(1j * phase[start:start+self.batch_size])
            intensity = self.propagate(field, self.fft_size)
            # Preserve physical irradiance units when input pitch or FFT changes.
            intensity = intensity * (self.pitch**2 * self.fft_size / (15**2 * 512))**2
            sampled = sum(torch.nn.functional.grid_sample(
                intensity[:, None], grid[None].expand(len(intensity), -1, -1, -1),
                align_corners=True, padding_mode="zeros")[:, 0] for grid in grids) / len(grids)
            predictions.append(sampled)
        return torch.cat(predictions)

    def update(self):
        started = time.monotonic_ns()
        torch = self.torch
        self.optimizer.zero_grad()
        pred = self.prediction()[self.train]
        measured = self.images[self.train]
        # Shape fit with cosine similarity; shared absolute gain is estimated
        # ONLY on training samples after fitting, never separately on test data.
        p, t = pred.flatten(1), measured.flatten(1)
        p = p - p.mean(1, keepdim=True)
        t = t - t.mean(1, keepdim=True)
        loss = (1 - (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1)).clamp_min(1e-8)).mean()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.d.clamp_(-2, 2)
            self.center.clamp_(0, 1920)
            self.log_width.clamp_(float(np.log(80)), float(np.log(800)))
        self.history.append(float(loss.detach()))
        self.iteration_records.append(dict(iteration=len(self.history), loss=self.history[-1],
                                           timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                                           monotonic_start_ns=started, monotonic_end_ns=time.monotonic_ns()))
        return self.history[-1]

    def run(self, iterations):
        # Optical-axis coordinates are measured in pixels, use a larger LR.
        for index in range(iterations):
            loss = self.update()
            if index % 25 == 0:
                print(f"fit {index}: loss={loss:.5f} d={self.d.item():.5f}", flush=True)
        return self.prediction().detach().numpy()


def load_observations(directory, camera_center=None, *, grid_size=128):
    """Read complete recorded phases and sample observations on a fixed CCD grid."""
    rows = json.loads((directory / "manifest.json").read_text())
    references = {r["branch"]: r for r in rows if r["label"] == "reference_start"}
    center = np.array([references["U"]["spot"][f"centroid_{k}"] for k in ("x", "y")]) if camera_center is None else np.asarray(camera_center)
    yy, xx = np.mgrid[-120:121:2, -120:121:2]
    camera_xy = np.stack((xx + round(center[0]), yy + round(center[1])), axis=-1)
    selected = [r for r in rows if r["family"] != "ramp" and r["settled"]["settled"]]
    phases, images = [], []
    for row in selected:
        with np.load(directory / row["file"]) as data:
            padded = np.zeros((1920, 1920), dtype=np.float32)
            padded[:1200] = data["panel_phase"]
            py, px = np.mgrid[:grid_size, :grid_size] * (1920. / grid_size)
            phases.append(map_coordinates(padded, [py, px], order=1))
            frame = data["raw_frames"].mean(0)
            images.append(frame[camera_xy[..., 1].astype(int), camera_xy[..., 0].astype(int)] / row["exposure_ms"])
    return selected, np.stack(phases), np.stack(images), camera_xy, references


def main():
    import torch
    from loguru import logger

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--residual-from", type=Path,
                        help="Fit seven residual modes starting from M3 in this JSON")
    parser.add_argument("--output", type=Path, help="New fit directory, keeping the source fit intact")
    args = parser.parse_args()
    logger.remove()
    torch.set_num_threads(4)
    out = args.directory
    destination = args.output or out
    if args.output:
        destination.mkdir(parents=True, exist_ok=False)
    elif (destination / "forward_fit.json").exists():
        parser.error("existing fit detected; supply a new --output directory")
    rows = json.loads((out / "manifest.json").read_text())
    config = json.loads((out / "run_config.json").read_text())
    matrix = ramp_matrix(rows)
    references = {r["branch"]: r for r in rows if r["label"] == "reference_start"}
    # Estimate illumination centre from chirp-induced displacement, diagnostic
    # approximation only; the amplitude-learning arm may adjust it on train data.
    shift = np.array([references["C"]["spot"][f"centroid_{k}"] - references["U"]["spot"][f"centroid_{k}"]
                      for k in ("x", "y")])
    gradient = np.linalg.solve(matrix, shift)
    q = (8e-6) ** 2 * config["best_diopter_m1"] / 1064e-9
    settings = config["settings"]
    amplitude_center = np.array([settings["center_x"], settings["center_y"]]) + gradient / q
    center = np.array([references["U"]["spot"][f"centroid_{k}"] for k in ("x", "y")])
    selected, phases, images, camera_xy, references = load_observations(out, center)
    train = np.array([r["split"] != "test" for r in selected])
    results = {"ramp_matrix": matrix.tolist(), "illumination_center_estimate": amplitude_center.tolist(),
               "camera_crop_center": center.tolist(), "sample_count": len(selected), "models": {}}
    arms = (("M0", False, 0., 0),
                                         ("M1_defocus", False, None, args.iterations),
                                         ("M3_illumination", True, None, args.iterations))
    if args.residual_from:
        if not args.output:
            parser.error("--residual-from requires a new --output directory")
        arms = (("M4_residual", True, None, args.iterations),)
    for name, learn, initial, budget in arms:
        model = PhysicalForwardFit(phases, images, camera_xy, matrix, config["best_diopter_m1"],
                                   train, amplitude_center, learn_amplitude=learn, initial_defocus=initial,
                                   learn_residual=bool(args.residual_from))
        with torch.no_grad():
            model.origin.copy_(torch.tensor(center, dtype=torch.float32))
            if args.residual_from:
                previous = json.loads(args.residual_from.read_text())["models"]["M3_illumination"]
                model.d.fill_(previous["defocus_diopter"])
                model.origin.copy_(torch.tensor(previous["origin"]))
                model.center.copy_(torch.tensor(previous["amplitude_center"]))
                model.log_width.copy_(torch.tensor(previous["amplitude_width"]).log())
        prediction = model.run(budget)
        design = np.stack((prediction[train].ravel(), np.ones(prediction[train].size)), axis=1)
        gain, background = nnls(design, images[train].ravel())[0]
        prediction = prediction * gain + background
        metrics = []
        for i, row in enumerate(selected):
            branch_center = np.array([references[row["branch"]]["spot"][f"centroid_{k}"] for k in ("x", "y")])
            mask = np.all(np.abs(camera_xy - branch_center) <= 30, axis=-1)
            optical = {}
            for label, array in (("measured", images[i]), ("predicted", prediction[i])):
                values = array[mask]
                optical[label + "_cv"] = float(values.std() / max(values.mean(), 1e-9))
                optical[label + "_ee_roi"] = float(values.sum() / max(array.sum(), 1e-9))
            metrics.append({"file": row["file"], "label": row["label"], "branch": row["branch"],
                            "split": row["split"], "family": row["family"],
                            **optical,
                            "corr": float(np.corrcoef(images[i].ravel(), prediction[i].ravel())[0, 1]),
                            "nrmse": float(np.sqrt(np.mean((images[i]-prediction[i])**2)) / max(images[i].max(), 1e-8))})
        result = {"defocus_diopter": model.d.item(), "origin": model.origin.detach().tolist(),
                  "amplitude_center": model.center.detach().tolist(), "amplitude_width": model.log_width.exp().detach().tolist(),
                  "gain": float(gain), "background": float(background), "history": model.history, "rows": metrics}
        result["residual_noll_radians"] = dict(zip(model.residual_modes, model.residual.detach().tolist()))
        with (destination / f"fit_iterations_{name}.pkl").open("wb") as stream:
            pickle.dump(model.iteration_records, stream, protocol=5)
        results["models"][name] = result
        np.savez_compressed(destination / f"prediction_{name}.npz", predicted=prediction, measured=images, camera_xy=camera_xy)
        (destination / "forward_fit.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(name, [(b, float(np.mean([r["corr"] for r in metrics if r["branch"] == b and r["split"] == "test"])))
                     for b in ("U", "C")], flush=True)


if __name__ == "__main__":
    main()
