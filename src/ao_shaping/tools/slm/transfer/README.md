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
