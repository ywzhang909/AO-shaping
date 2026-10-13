---
title: save_optimization_debug_artifacts / save_recorder_debug_artifacts（调试产物落盘）
created: 2026-10-13
updated: 2026-10-13
type: function
tags: [io,debug]
source: src/ao_shaping/utils/io/file.py:739
---

## 签名

```python
save_optimization_debug_artifacts(records, save_dir, saved_file_name, min_epoch, min_metric, best_coeff_key, init_wavefront, opt_wavefront, init_title, opt_title, plot_params_note, data, … (+7 个)) → `Path | None`
```

## docstring（原文）

```
Emit the standard debug artifacts for a completed optimisation run.

    Two mutually exclusive modes:

    * **Wavefront mode** (default): ``records`` is a :class:`Recorder` object;
      writes the 2×2 debug PNG + compressed dataframe. This is the
      historical behaviour used by ``ga_zernike_runner``,
      ``greedy_zernike_runner`` and ``rms_zernike_runner``.
    * **Data mode**: ``data`` is a ``{epoch: record}`` dict; writes a 2×2
      PNG (objective history / best coefficients / first & last image), a
      pickled copy of ``data`` and a JSON sidecar of ``json_payload``.
      Used by ``slm_shaping_runner``.

    Args:
        records:           Recorder object with ``get_sublist()``,
                           ``get_best_iter()``, ``first``, and
                           ``save_dataframe``. (wavefront mode)
        save_dir:          Directory in which to write output files.
        saved_file_name:   UUID filename prefix (no extension).
        min_epoch:         Epoch index of the best result.
        min_metric:        Primary metric value at *min_epoch*
                           (RMS, PIB, …).
        best_coeff_key:    Dictionary key for the coefficient vector
                           in the best-epoch record (e.g. ``"_c"`` or ``"_v"``).
        init_wavefront:    Wavefront array for the initial state.
        opt_wavefront:     Wavefront array for the optimised state.
        init_title:        Colorbar / panel title for the initial WF.
        opt_title:         Colorbar / panel title for the optimised WF.
        plot_params_note:  Optional suffix appended to the *voltages* panel
                           title (e.g. ``"epoch=200"``).
        data:              ``{epoch: record}`` dict of scalar/array fields.
                           (data mode; mutually exclusive with ``records``)
        png_path:          Output path for the figure. (data mode)
        pkl_path:          Output path for the pickled ``data``. (data mode)
        json_path:         Output path for the JSON ``json_payload``. (data mode)
        title:             Figure title. (data mode)
        json_payload:      Dict serialised to ``json_path``. (data mode)
        top_right_key:     Record key rendered in the top-right panel. When set,
                           that key's last-epoch array is shown instead of the
                           ``_c`` bar chart. (data mode)
        top_right_title:   Panel title override for ``top_right_key``. (data mode)

    Returns:
        ``png_path`` in data mode, ``None`` in wavefront mode.
```

## 直接调用（出边）

- `_save_data_mode_debug_artifacts`

## 谁调用它（入边）

- [[commands/rms-zernike|`run`]]

## 相关页面

- [[entry/cli]]
