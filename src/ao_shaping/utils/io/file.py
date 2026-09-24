from typing import Any, Literal
import os
import re
import json
import pickle
from loguru import logger

import uuid
from pathlib import Path
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project root directory (workspace root)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

# Image file extensions to search
_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def find_cell_image(base_dir: str | Path, ip_group: int, seq: int) -> Path | None:
    """根据 IP组+序号 在目录中查找对应的单元格图片。

    支持两种命名格式:
    - 旧格式: {ip}-{seq:03d}.png
    - 新格式: {ip}-{seq:03d}_cx*_cy*.png (含质心坐标后缀)

    目录结构: {base_dir}/192.168.0.{ip}/{ip}-{seq:03d}*.png

    Args:
        base_dir: 图片根目录 (如 data/md_test/md_img 或 data/md_test/md_img-100v_processed/diff)
        ip_group: IP 组号 (101~126)
        seq: 序号 (0~49)

    Returns:
        匹配的图片 Path，未找到则返回 None
    """
    base = Path(base_dir)
    ip_dir = base / f"192.168.0.{ip_group}"
    if not ip_dir.exists():
        return None

    prefix = f"192.168.0.{ip_group}-{seq:03d}"
    for f in ip_dir.iterdir():
        if f.is_file() and f.stem.startswith(prefix) and f.suffix.lower() in _IMG_EXTENSIONS:
            return f
    return None


def gen_file_path_inc(dir: str | Path, postfix: str = ""):
    if isinstance(dir, str):
        dir = Path(dir)
    if not dir.exists():
        dir.mkdir(parents=True)
    fname = os.listdir(dir)
    if postfix:
        fname = len([_ for _ in fname if _.endswith(postfix)]) + 1
    else:
        fname = len(fname) + 1

    if not postfix:  # make dir
        path = dir.joinpath(str(fname))
        if not postfix and not os.path.exists(path):
            os.makedirs(path)
    else:
        path = dir.joinpath(str(fname)).with_suffix(postfix)
    return path


def gen_file_path_uuid(dir: str | Path, postfix: str = ""):
    # generate file path with uuid
    if isinstance(dir, str):
        dir = Path(dir)
    if not dir.exists():
        dir.mkdir(parents=True)
    fname = str(uuid.uuid4())
    path = dir.joinpath(fname)
    if postfix:
        path = path.with_suffix(postfix if postfix.startswith(".") else f".{postfix}")
    return path


def gen_date_str():
    # generate date string
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def gen_date_dir(base_dir: str | Path = "data"):
    # generate date dir
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
    date_str = gen_date_str()
    date_dir = base_dir.joinpath(date_str)
    if not date_dir.exists():
        date_dir.mkdir(parents=True)
    return date_dir


def build_debug_save_paths(
    root_dir,
    context_subdir: Path | str,
) -> tuple[Path, Path]:
    """Build the date-stamped save directory + file-stem prefix used by all runners.

    Consistently mirrors the pattern::

        save_dir = gen_date_dir(Path(root_dir) / context_subdir)
        saved_file_name = save_dir / f"{context_subdir}_{save_dir.name}"

    Args:
        root_dir:     Top-level run-output root (e.g. ``"data"``).
        context_subdir:
            The runner-specific sub-path *relative to root_dir*.
            May be a :class:`~pathlib.Path` or a plain string, e.g.::

                ``"flatten_zernike"``
                ``"flatten_voltages"``
                ``Path("pipeline")``

    Returns:
        A 2-tuple ``(save_dir, saved_file_name)`` where ``save_dir`` is the
        full :class:`~pathlib.Path` to the date-stamped save directory and
        ``saved_file_name`` is a stem :class:`~pathlib.Path` (no extension)
        inside it, used to derive ``.png`` / ``.pkl`` / ``.json`` / ``.zip``
        siblings.
    """
    save_dir = gen_date_dir(Path(root_dir) / context_subdir)
    saved_file_name = save_dir / f"{context_subdir}_{save_dir.name}"
    return save_dir, saved_file_name


# 在当天data下的flatten_voltages文件夹找出rms最小的文件，读取电压值，如果没有则返回全0
def get_init_V_by_rms(date: str = ""):
    data_path = (
        f"data/flatten_voltages/{date}"
        if date
        else f"data/flatten_voltages/{datetime.now().strftime('%Y%m%d')}"
    )

    def get_rms(file_name):
        regex_pattern = r"rms-(\d+\.\d+)\.csv"
        match = re.search(regex_pattern, file_name)
        if match:
            return float(match.group(1))
        return np.nan

    try:
        file_list = glob(f"{data_path}/rms-*.csv")
        if not file_list:
            raise FileNotFoundError
        min_rms = min([get_rms(f) for f in file_list if not np.isnan(get_rms(f))])
        init_V = np.loadtxt(f"{data_path}/rms-{min_rms:.3f}.csv")
        logger.info(f"init_V by rms {min_rms:.3f}")
    except FileNotFoundError:
        init_V = np.zeros(64)
        logger.info(f"init_V by rms in {data_path} not found, return 0")
    return init_V


def get_init_V_by_energy(date: str = ""):
    data_path = (
        f"data/flatten_voltages/{date}"
        if date
        else f"data/flatten_voltages/{datetime.now().strftime('%Y%m%d')}"
    )

    def get_energy(file_name):
        regex_pattern = r"to_load_V-(\d+\.\d+)\.csv"
        match = re.search(regex_pattern, file_name)
        if match:
            return float(match.group(1))
        return np.nan

    try:
        max_energy = max(
            [
                get_energy(f)
                for f in glob(f"{data_path}/to_load_V-*.csv")
                if not np.isnan(get_energy(f))
            ]
        )
        init_V = np.loadtxt(f"{data_path}/to_load_V-{max_energy:.3f}.csv")
        logger.info(f"init_V by energy {max_energy:.3f}")
    except FileNotFoundError or ValueError:
        init_V = np.zeros(64)
        logger.info(f"init_V by energy @ {data_path} not found, return 0")
    return init_V


def save_history(
    history: pd.DataFrame | list[dict[str, Any]], file_path: str | Path = None,
    sidecar_dir: str | Path | None = None,
):
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame(history)
    if file_path is not None:
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = file_path.with_suffix('.csv')
        history.to_csv(csv_path, index=False)
        logger.info(f"History saved to {csv_path}")
        if sidecar_dir is not None:
            _save_array_sidecars(history, sidecar_dir, file_path)


def _save_array_sidecars(
    history: pd.DataFrame,
    sidecar_dir: str | Path,
    base_path: str | Path,
):
    """Extract numpy-array columns from *history* and save each as an .npy sidecar.

    Columns whose values are numpy arrays are identified by inspecting the
    first non-null entry.  One ``.npy`` file is written per epoch:

        <base_name>_<col>_<epoch:04d>.npy

    where *base_name* is the stem of *base_path* (e.g. ``"myrun"`` from
    ``myrun.csv``).
    """
    sidecar_dir = Path(sidecar_dir)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    base = Path(base_path).stem if base_path else "history"

    array_cols: list[str] = []
    for col in history.columns:
        vals = history[col].dropna()
        if len(vals) > 0 and isinstance(vals.iloc[0], np.ndarray):
            array_cols.append(col)

    if not array_cols:
        return

    npy_paths: list[Path] = []
    for epoch_idx, row in history.iterrows():
        for col in array_cols:
            arr = row[col]
            if isinstance(arr, np.ndarray):
                fname = f"{base}_{col}_{epoch_idx:04d}.npy"
                path = sidecar_dir / fname
                np.save(path, arr)
                npy_paths.append(path)

    logger.info(
        f"Saved {len(npy_paths)} array sidecar files "
        f"for columns {array_cols} to {sidecar_dir}"
    )


def save_history_hdf5(
    history: "pd.DataFrame | list[dict[str, Any]] | Recorder",
    file_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Export an optimization history to a single HDF5 file.

    把 (Recorder / 行列表 / DataFrame) 历史导出为 HDF5, 每轮的标量列、数组列
    (系数/图像/梯度/完整相位) 集中保存, 便于离线逐轮分析。结构:

    .. code-block:: text

        <stem>.h5
        ├── /metadata                  metadata 标量/字符串属性
        ├── /scalars/<col>             每个标量列一条 1-D 数据集
        └── /epochs/<epoch:04d>/       每轮一个组
            └── <col>                  每列数组 (ndarray) 数据集

    数组列与标量列依据首条非空值识别 (np.ndarray/list/tuple → 数组列)。
    逐轮分组而非堆叠数据集, 以容纳长度/形状不同的列 (如 ``_c`` 与
    ``_img`` 形状不同)。``h5py`` 在函数内延迟导入, 保持叶子层导入轻量。

    Args:
        history: 优化历史 —— :class:`Recorder`、行字典列表或 DataFrame。
        file_path: 目标路径 (后缀自动替换为 ``.h5``)。
        metadata: 可选的标量/字符串元数据, 写入 ``/metadata`` 组属性。

    Returns:
        实际写入的 ``.h5`` 路径。

    Raises:
        ValueError: ``history`` 为空。
    """
    import h5py

    if isinstance(history, Recorder):
        rows: list[dict[str, Any]] = list(history.history)
    elif isinstance(history, pd.DataFrame):
        rows = [row.to_dict() for _, row in history.iterrows()]
    else:
        rows = list(history)
    if not rows:
        raise ValueError("history is empty - nothing to export")

    path = Path(file_path).with_suffix(".h5")
    path.parent.mkdir(parents=True, exist_ok=True)

    all_cols = sorted({k for row in rows for k in row.keys()})
    numeric_cols: list[str] = []
    str_cols: list[str] = []
    array_cols: list[str] = []
    for col in all_cols:
        vals = [row[col] for row in rows if col in row and row[col] is not None]
        if vals and all(isinstance(v, (np.ndarray, list, tuple)) for v in vals):
            array_cols.append(col)
        elif vals and all(isinstance(v, str) for v in vals):
            str_cols.append(col)
        elif vals and all(
            isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_))
            for v in vals
        ):
            numeric_cols.append(col)
        elif vals:
            logger.warning(
                "skip mixed/non-scalar column {} in HDF5 export", col
            )

    with h5py.File(path, "w") as f:
        if metadata:
            meta = f.create_group("metadata")
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    meta.attrs[key] = value
                elif isinstance(value, (np.integer, np.floating, np.bool_)):
                    meta.attrs[key] = value.item()
                elif value is None:
                    meta.attrs[key] = ""

        scalars = f.create_group("scalars")
        for col in numeric_cols:
            scalars.create_dataset(
                col,
                data=np.asarray(
                    [row.get(col, np.nan) for row in rows], dtype=float
                ),
            )
        for col in str_cols:
            scalars.create_dataset(
                col,
                data=np.asarray(
                    [row.get(col, "") for row in rows],
                    dtype=h5py.string_dtype(encoding="utf-8"),
                ),
            )

        epochs = f.create_group("epochs")
        for idx, row in enumerate(rows):
            g = epochs.create_group(f"{int(row.get('_id', idx)):04d}")
            for col in array_cols:
                if col not in row or row[col] is None:
                    continue
                g.create_dataset(col, data=np.asarray(row[col]))

    logger.info("History exported to {}", path)
    return path


class Recorder:
    def __init__(self, mark: str = "J", mode: Literal["max", "min"] = "max"):
        self.mark = mark
        self.mode = mode
        self.history = list()

        self._all_columns = set()
        self._postprocess_funcs: dict[str, callable] = {}

    def append(self, record: dict):
        assert self.mark in record, f"mark {self.mark} not in record {record}"
        if "_id" not in record:
            record["_id"] = len(self.history)
        self.history.append(record)
        self._all_columns.update(record.keys())

    def postprocess_feature(self, feature_name: str, func: callable, column: str = ""):
        """为 history 添加后处理特征列。

        Args:
            feature_name: 特征名称，可作为 get_best_* 系列函数的输入
            func: 计算函数，签名为 func(row_dict) -> value
            column: 保存到 DataFrame 的列名，默认为 feature_name
        """
        if not column:
            column = feature_name
        self._postprocess_funcs[feature_name] = (func, column)
        self._all_columns.add(column)

    def _apply_postprocess_to_record(self, record: dict) -> dict:
        """对单条记录应用所有已注册的后处理函数。"""
        result = dict(record)
        for feature_name, (func, column) in self._postprocess_funcs.items():
            if column not in result:
                try:
                    result[column] = func(result)
                except Exception as e:
                    logger.warning(f"postprocess_feature '{feature_name}' failed: {e}")
                    result[column] = None
        return result

    def _ensure_postprocess_applied(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保 DataFrame 包含所有后处理列。"""
        for feature_name, (func, column) in self._postprocess_funcs.items():
            if column not in df.columns:

                def _safe_apply(row, f=func):
                    try:
                        return f(row.to_dict())
                    except Exception as e:
                        logger.warning(f"postprocess_feature failed: {e}")
                        return None

                df[column] = df.apply(_safe_apply, axis=1)
        return df

    @property
    def dataframe(self):
        df = pd.DataFrame(self.history)
        return self._ensure_postprocess_applied(df)

    def save_dataframe(self, filename: str | Path, sidecar_dir: str | Path | None = None, **kwargs):
        df = self.dataframe
        save_history(df, filename, sidecar_dir=sidecar_dir)
        return df

    def save_array_sidecars(self, sidecar_dir: str | Path) -> None:
        """Save /_wavefront/ and /_phase/ columns as .npy sidecar files.

        One ``.npy`` file written per epoch and per column:
        ``<sidecar_dir>/<base>_<col>_<epoch:04d>.npy`` where
        *base* is ``recorder.mark``.
        """
        _save_array_sidecars(self.dataframe, sidecar_dir, self.mark)

    def save_best(
        self, saved_dir: str | Path, target: str, process_fn=lambda x: x, **kwargs
    ):
        target_value, (index, value) = self.get_best_target(target)

        if isinstance(saved_dir, str):
            saved_dir = Path(saved_dir)
        saved_dir.mkdir(parents=True, exist_ok=True)

        target_value = process_fn(target_value)
        if isinstance(target_value, np.ndarray) and target_value.ndim == 1:  # 1D array
            save_file = saved_dir / f"{self.mark}-{value:.3f}.csv"
            np.savetxt(save_file, target_value, **kwargs)
        elif (
            isinstance(target_value, np.ndarray) and target_value.ndim == 2
        ):  # 2D array
            save_file = saved_dir / f"{self.mark}-{value:.3f}.png"
            plt.imshow(target_value, **kwargs)
            plt.savefig(save_file)
            plt.close()
        else:
            raise ValueError(
                f"target_value {target_value} has invalid shape {target_value.shape}"
            )
        logger.info(f"{self.mark}@{index}->{value:.3f} saved to {save_file}")
        return target_value, value

    def plot(self, target: str, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(self.history[target], self.history[self.mark])
        ax.set_xlabel(target)
        ax.set_ylabel(self.mark)
        ax.set_title(f"{self.mark} vs {self.target}")
        return ax

    def __len__(self):
        return len(self.history)

    def __getitem__(self, index):
        assert index < len(self.history), (
            f"index {index} out of range {len(self.history)}"
        )
        if hasattr(self.history, 'iloc'):
            return self.history.iloc[index]
        return self.history[index]

    def __add__(self, other: "Recorder"):
        assert self.mark == other.mark, (
            "mark must be the same"
        )
        self.history.extend(other.history)
        return self

    @property
    def columns(self):
        return list(self._all_columns)

    @property
    def last(self):
        return self.history[-1]

    @property
    def first(self):
        return self.history[0]

    @property
    def last_info_dict(self) -> dict[str, Any]:
        info_dict = self.last
        return {k: v for k, v in info_dict.items() if not k.startswith("_")}

    def get_best_iter(self, mark: str = ""):
        mark = mark or self.mark
        res_df = self.dataframe
        target_id = (
            res_df[mark].argmax() if self.mode == "max" else res_df[mark].argmin()
        )
        return res_df.iloc[target_id], (target_id, res_df.iloc[target_id][mark])

    def get_best_target(self, target):
        if target not in self.columns:
            target = "_" + target
            if target not in self.columns:
                raise ValueError(f"target {target} not in columns {self.columns}")
        target_iter, (index, value) = self.get_best_iter()
        return target_iter[target], (index, value)

    def get_sublist(self, columns: list[str] | str | None = ""):
        if not columns:
            columns = self.mark
        if isinstance(columns, str):
            return [l.get(columns, np.nan) for l in self.history]
        else:
            return [{k: v for k, v in l.items() if k in columns} for l in self.history]


# ---------------------------------------------------------------------------
# Optimisation-debug visualisation block
# ---------------------------------------------------------------------------

_DATA_MODE_OBJECTIVE_KEYS = ("pib", "radiu", "avg_radiu")


def _infer_objective_key(data: dict[int, dict]) -> str | None:
    """Return the objective column name present in the first data record."""
    first = next(iter(data.values()), {})
    for key in _DATA_MODE_OBJECTIVE_KEYS:
        if key in first:
            return key
    return None


def _save_data_mode_debug_artifacts(
    data: dict[int, dict],
    png_path: Path,
    pkl_path: Path,
    json_path: Path,
    title: str,
    json_payload: dict | None,
) -> Path:
    """Write PNG / pkl / json debug artifacts for a ``{epoch: record}`` dict.

    Figure layout (2×2): objective history line plot (top-left), best
    coefficient bar chart (top-right), first ``_img`` (bottom-left), last
    ``_img`` (bottom-right).
    """
    epochs = sorted(data.keys())

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    obj_key = _infer_objective_key(data)
    if obj_key is not None:
        xs = [e for e in epochs if obj_key in data[e]]
        ys = [data[e][obj_key] for e in xs]
        ax[0, 0].plot(xs, ys)
        ax[0, 0].set_xlabel("epoch")
        ax[0, 0].set_ylabel(obj_key)
    ax[0, 0].set_title(title)

    c_arr = None
    for e in reversed(epochs):
        if "_c" in data[e]:
            c_arr = np.asarray(data[e]["_c"])
            break
    ax[0, 1].set_title("best coefficients")
    if c_arr is not None:
        ax[0, 1].bar(range(len(c_arr)), c_arr)
    else:
        ax[0, 1].text(0.5, 0.5, "no _c", ha="center", va="center")

    imgs = [data[e]["_img"] for e in epochs if "_img" in data[e]]
    for ax_i, label, img in (
        (ax[1, 0], "first _img", imgs[0] if imgs else None),
        (ax[1, 1], "last _img", imgs[-1] if imgs else None),
    ):
        ax_i.set_title(label)
        if img is not None:
            ax_i.imshow(np.asarray(img))
        else:
            ax_i.text(0.5, 0.5, "no _img", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=4)

    return png_path


def save_optimization_debug_artifacts(
    records : Recorder | None = None,
    save_dir: Path | None = None,
    saved_file_name: Path | None = None,
    min_epoch: int | None = None,
    min_metric: float | None = None,
    best_coeff_key: str | None = None,
    init_wavefront=None,
    opt_wavefront=None,
    init_title: str = "init wavefront",
    opt_title: str = "opt wavefront",
    plot_params_note: str | None = None,
    *,
    data: dict[int, dict] | None = None,
    png_path: Path | None = None,
    pkl_path: Path | None = None,
    json_path: Path | None = None,
    title: str = "",
    json_payload: dict | None = None,
) -> Path | None:
    """Emit the standard debug artifacts for a completed optimisation run.

    Two mutually exclusive modes:

    * **Wavefront mode** (default): ``records`` is a :class:`Recorder` object;
      writes the 2×2 debug PNG + compressed dataframe. This is the
      historical behaviour used by ``ga_zernike_runner``,
      ``greedy_zernike_runner`` and ``rms_zernike_runner``.
    * **Data mode**: ``data`` is a ``{epoch: record}`` dict; writes a 2×2
      PNG (objective history / best coefficients / first & last image), a
      pickled copy of ``data`` and a JSON sidecar of ``json_payload``.
      Used by ``slm_pib_runner``.

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

    Returns:
        ``png_path`` in data mode, ``None`` in wavefront mode.
    """
    if data is not None:
        if records is not None:
            raise ValueError(
                "pass either records (wavefront mode) or data (data mode), not both"
            )
        assert png_path is not None and pkl_path is not None and json_path is not None
        return _save_data_mode_debug_artifacts(
            data=data,
            png_path=png_path,
            pkl_path=pkl_path,
            json_path=json_path,
            title=title,
            json_payload=json_payload,
        )

    # --- wavefront mode (historical behaviour) ---
    from ao_shaping.utils.image.display import make_debug_wavefront_ax_plots, plot_funcs

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    rms_values = records.get_sublist()
    plot_funcs["rms_history"](rms_values, ax[0, 0], min_epoch, min_metric)

    best_coeffs = records.get_best_iter()[0][best_coeff_key]
    title_suffix = (
        f"{plot_params_note}" if plot_params_note
        else f"{min_metric:.3f} @ epoch {min_epoch}"
    )
    plot_funcs["voltages"](best_coeffs, ax[0, 1], title_suffix)

    make_debug_wavefront_ax_plots(ax[1], init_wavefront, opt_wavefront,
                                   init_title=init_title, opt_title=opt_title)

    plt.tight_layout()
    plt.savefig(saved_file_name.with_suffix(".png"))
    plt.close()

    records.save_dataframe(saved_file_name.with_suffix(".zip"),
                          compression="zip")
    return None


# ---------------------------------------------------------------------------
# Generic recorder → {epoch: record} debug-artifact writer
# ---------------------------------------------------------------------------

def save_recorder_debug_artifacts(
    res: Recorder,
    root_dir: str,
    subdir_prefix: str,
    *,
    scalar_keys: tuple[str, ...] = (),
    objective_keys: tuple[str, ...] = (),
    img_keys: tuple[str, ...] = ("_img",),
    d1_keys: tuple[str, ...] = (),
    d2_keys: tuple[str, ...] = (),
    json_payload: dict | None = None,
    title: str = "",
) -> Path:
    """Convert a :class:`Recorder` into the data-mode debug artifacts.

    This is the shared backend for ``slm_pib_runner._save_debug_artifacts``
    and ``slm_gsnet_runner._save_debug_artifacts``: it walks ``res.history``,
    extracts the configured key sets into a ``{epoch: record}`` dict and
    delegates to :func:`save_optimization_debug_artifacts` (data mode).

    Args:
        res:             Recorder object whose ``.history`` is a list of dicts.
        root_dir:        Top-level run-output root (e.g. ``"data"``).
        subdir_prefix:   Runner-specific sub-path relative to ``root_dir``
                         (e.g. ``"slm_pib_pib"``). A timestamp is appended.
        scalar_keys:     Keys whose values are cast to ``float``.
        objective_keys:  Like ``scalar_keys`` but searched first; merged into
                         the scalar set.
        img_keys:        Keys whose values are 2D image arrays.
        d1_keys:         Keys whose values are 1D coefficient arrays.
        d2_keys:         Keys whose values are 2D gradient arrays.
        json_payload:    Dict serialised to the ``.json`` sidecar.
        title:           Figure title.

    Returns:
        The ``.png`` path written.
    """
    from datetime import datetime as _dt

    save_dir, saved_file_name = build_debug_save_paths(
        os.path.join(root_dir, "debug"),
        f"{subdir_prefix}_{_dt.now():%Y%m%d_%H%M%S}",
    )
    png_path = saved_file_name.with_suffix(".png")
    pkl_path = saved_file_name.with_suffix(".pkl")
    json_path = saved_file_name.with_suffix(".json")

    data: dict[int, dict] = {}
    for rec in res.history:
        item: dict[str, Any] = {}
        for k in tuple(scalar_keys) + tuple(objective_keys):
            if k in rec:
                item[k] = float(rec[k])
        for k in img_keys:
            if k in rec:
                item[k] = np.asarray(rec[k])
        for k in d1_keys:
            if k in rec:
                item[k] = np.asarray(rec[k], dtype=float)
        for k in d2_keys:
            if k in rec:
                item[k] = np.asarray(rec[k])
        data[int(rec["_epoch"])] = item

    save_optimization_debug_artifacts(
        data=data,
        png_path=png_path,
        pkl_path=pkl_path,
        json_path=json_path,
        title=title,
        json_payload=json_payload,
    )
    return png_path


class DeviceConfigManager:
    """通用设备配置管理器
    
    管理设备的JSON配置文件加载和保存，支持所有设备类型。
    配置文件按设备序列号存储，路径: <config_dir>/{device_type}/{serial_number}.json
    
    支持默认启动参数，可在配置目录下放置 defaults.json 作为全局默认配置。
    """

    def __init__(self, config_dir: str | Path, device_type: str = ""):
        """初始化配置管理器
        
        Args:
            config_dir: 配置文件根目录路径
            device_type: 设备类型标识（如 'slm', 'dm', 'ccd' 等）
        """
        self.config_dir = Path(config_dir)
        self.device_type = device_type
        self.device_config_dir = self.config_dir / device_type if device_type else self.config_dir
        self.device_config_dir.mkdir(parents=True, exist_ok=True)

        # 加载默认配置
        self._default_config = self._load_default_config()

    def _load_default_config(self) -> dict:
        """加载全局默认配置文件 defaults.json"""
        default_file = self.config_dir / "defaults.json"
        if default_file.exists():
            try:
                with open(default_file, encoding="utf-8") as f:
                    defaults = json.load(f)
                # 返回对应设备类型的默认配置
                if self.device_type and self.device_type in defaults:
                    return defaults[self.device_type]
                return defaults
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"读取默认配置失败: {e}")
        return {}

    def _get_config_file(self, serial: str) -> Path:
        """获取配置文件路径"""
        return self.device_config_dir / f"{serial}.json"

    def load_config(self, serial: str) -> dict:
        """根据序列号加载JSON配置文件
        
        配置文件路径: <config_dir>/{device_type}/{serial}.json
        如果文件不存在，返回默认配置（如果已设置）
        
        Args:
            serial: 设备序列号
            
        Returns:
            配置字典；合并默认配置和设备特定配置
        """
        config_file = self._get_config_file(serial)

        # 从默认配置开始
        config = dict(self._default_config)

        if not config_file.exists():
            logger.info(f"未找到{self.device_type}设备({serial})配置文件，使用默认参数")
            return config

        try:
            with open(config_file, encoding="utf-8") as f:
                device_config = json.load(f)
            # 合并设备特定配置（覆盖默认值）
            config.update(device_config)
            logger.info(f"已加载{self.device_type}设备({serial})配置文件")
            return config
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"读取配置文件失败: {e}，使用默认参数")
            return config

    def save_config(self, serial: str, config: dict) -> None:
        """将配置保存到JSON文件
        
        配置文件路径: <config_dir>/{device_type}/{serial}.json
        
        Args:
            serial: 设备序列号
            config: 配置字典
        """
        config_file = self._get_config_file(serial)

        # 确保配置中包含序列号
        config_with_serial = dict(config)
        config_with_serial["serial_number"] = serial

        try:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_with_serial, f, indent=2, ensure_ascii=False)
            logger.info(f"配置已保存: {config_file}")
        except OSError as e:
            logger.error(f"保存配置失败: {e}")

    def config_exists(self, serial: str) -> bool:
        """检查指定序列号的配置文件是否存在
        
        Args:
            serial: 设备序列号
            
        Returns:
            配置文件是否存在
        """
        return self._get_config_file(serial).exists()

    def delete_config(self, serial: str) -> bool:
        """删除指定序列号的配置文件
        
        Args:
            serial: 设备序列号
            
        Returns:
            是否成功删除
        """
        config_file = self._get_config_file(serial)
        try:
            if config_file.exists():
                config_file.unlink()
                logger.info(f"配置已删除: {config_file}")
                return True
            return False
        except OSError as e:
            logger.error(f"删除配置失败: {e}")
            return False

    def list_configs(self) -> list[str]:
        """列出所有已保存的配置文件对应的序列号
        
        Returns:
            序列号列表
        """
        try:
            return [f.stem for f in self.device_config_dir.glob("*.json")]
        except OSError:
            return []

    def set_default_config(self, defaults: dict) -> None:
        """设置默认配置（运行时）
        
        Args:
            defaults: 默认配置字典
        """
        self._default_config = dict(defaults)

    def save_default_config(self, defaults: dict) -> None:
        """保存默认配置到文件 defaults.json
        
        Args:
            defaults: 默认配置字典，可按设备类型组织
                      如: {'slm': {'wavelength': 1064}, 'dm': {'voltages': [0]*64}}
        """
        default_file = self.config_dir / "defaults.json"
        try:
            with open(default_file, "w", encoding="utf-8") as f:
                json.dump(defaults, f, indent=2, ensure_ascii=False)
            logger.info(f"默认配置已保存: {default_file}")
            # 重新加载
            self._default_config = self._load_default_config()
        except OSError as e:
            logger.error(f"保存默认配置失败: {e}")


# 向后兼容：SLMConfigManager 作为 DeviceConfigManager 的别名
class SLMConfigManager(DeviceConfigManager):
    """SLM设备配置管理器（DeviceConfigManager的别名，用于向后兼容）"""

    def __init__(self, config_dir: str | Path):
        super().__init__(config_dir, device_type="slm")
