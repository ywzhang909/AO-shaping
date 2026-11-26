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

`snapshot.py` and `branch_fit.py` implement the #254 T3/T4 offline toolchain
(#254 T2 already lives in `splits.py` + `adaptation.validation_quality`).
`snapshot freeze` turns recorded capture directories into an immutable,
hash-addressed three-tier snapshot (`--plan-only` prints the pre-registration
plan first); `snapshot verify` re-hashes every file, with `--root` to relocate
a moved corpus. `branch_fit freeze --spec <json> --manifest <snapshot.json>`
trains the pre-registered branches under one shared budget (only model knobs
from `BRANCH_OVERRIDE_WHITELIST` may differ per branch), scores each branch on
the `validation` tier only, and writes an immutable `candidates_frozen.json`;
`branch_fit score-final` then scores the sealed `final_test` tier exactly once
per frozen candidate (audit-logged, second attempt refused). Branch sets are
never hardcoded here: the spec file is the pre-registration, and the plan
document that defines it lives beside the corpus on the capture machine.
Both modules are device-free and importable without torch at module scope;
`branch_fit` calls into `fit.fit` and needs torch only at run time.

`uncertainty.py` is an experimental, device-free diagnostic: `disagreement`
ranks saved predictions by mean-normalised ensemble RMS spread, `select_diverse`
adds a wrapped-command diversity constraint, and `evaluate` reports paired
shape error on a fixed camera grid. It also offers an offline CLI:
`python -m ao_shaping.tools.slm.transfer.uncertainty rank --input <npz>
--count N --out <json>` and the matching `evaluate` subcommand. The ranking
score is not calibrated to a probability and is not an acceptance gate; always
validate it against residuals, family quotas and a random control. No function
in this module opens a device.
