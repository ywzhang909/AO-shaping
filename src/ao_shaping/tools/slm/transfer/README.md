# Physical forward transfer workflow

Run `python -m ao_shaping.tools.slm.transfer --help` for the integrated CLI.
The [operation guide](../../../../../docs/slm/slm_transfer_workflow.md)
documents planning, physical execution, offline synthesis, archive replay and
raw-data audit. Importing any module in this package does not open devices.

`workflow.TransferSession` and `workflow.run_session` provide the Python API.
Device actions occur only in capture/validation methods explicitly invoked by
the caller. Shared numerical code lives in `algorithm.signal_processing` and
`utils.image`; this package does not import the repository's `scripts` tree.

Existing scripts are compatibility wrappers. Keep mathematical implementation
in source modules and report rendering in reporting scripts. Preserve raw
phases, displayed grayscale, camera grids/masks, exposure readbacks and clocks.

Warm starts default to `--adaptation auto`: fit illumination position and shared
radiometry first, then fit all optical parameters if validation warrants it.
The unchanged optical baseline remains a candidate. `fit/adaptation.json`
records the selection; paired CCD acceptance remains independent of this choice.
Single-shape runs automatically skip the three-shape response/blur calibration.
Use `simulate --out <new-directory>` to create deterministic known-truth linear
camera data for device-free fitting and synthesis with `run --offline`.

To continue an accepted shaping result without GS, specify `--seed-run <run>`.
Use `--seed-policy transfer` to compensate newly fitted static aberrations and
the measured carrier mapping; `strict` requires identical grids and targets.
With `--skip-calibration`, this warm start goes straight from current-path
capture and fitting to anchored CCD blocks. It records `gs_iterations: 0` and
per-shape phase-transfer provenance in the validation configuration.
