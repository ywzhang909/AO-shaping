from typing import Any, Literal
import os
import re
import json
from loguru import logger

import uuid
from pathlib import Path
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project root directory (workspace root)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent


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
