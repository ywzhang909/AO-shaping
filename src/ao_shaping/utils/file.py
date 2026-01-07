from typing import Any, Literal
import os
import re
from loguru import logger

import uuid
from pathlib import Path
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gen_file_path_inc(dir: str|Path, postfix: str = ''):
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

def gen_file_path_uuid(dir: str|Path, postfix: str = ''):
    # generate file path with uuid
    if isinstance(dir, str):
        dir = Path(dir)
    if not dir.exists():
        dir.mkdir(parents=True)
    fname = str(uuid.uuid4())
    path = dir.joinpath(fname)
    if postfix:
        path = path.with_suffix(postfix if postfix.startswith('.') else f'.{postfix}')
    return path

def gen_date_str():
    # generate date string
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def gen_date_dir(base_dir: str|Path = 'data'):
    # generate date dir
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
    date_str = gen_date_str()
    date_dir = base_dir.joinpath(date_str)
    if not date_dir.exists():
        date_dir.mkdir(parents=True)
    return date_dir

# 在当天data下的flatten_voltages文件夹找出rms最小的文件，读取电压值，如果没有则返回全0
def get_init_V_by_rms(date:str = ''):
    data_path = f"data/flatten_voltages/{date}" if date else f"data/flatten_voltages/{datetime.now().strftime('%Y%m%d')}"
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

def get_init_V_by_energy(date:str = ''):
    data_path = f"data/flatten_voltages/{date}" if date else f"data/flatten_voltages/{datetime.now().strftime('%Y%m%d')}"
    def get_energy(file_name):
        regex_pattern = r"to_load_V-(\d+\.\d+)\.csv"
        match = re.search(regex_pattern, file_name)
        if match:
            return float(match.group(1))
        return np.nan
    
    try:
        max_energy = max([get_energy(f) for f in glob(f"{data_path}/to_load_V-*.csv") if not np.isnan(get_energy(f))])
        init_V = np.loadtxt(f"{data_path}/to_load_V-{max_energy:.3f}.csv")
        logger.info(f"init_V by energy {max_energy:.3f}")
    except FileNotFoundError or ValueError:
        init_V = np.zeros(64)
        logger.info(f"init_V by energy @ {data_path} not found, return 0")
    return init_V

def save_history(history:pd.DataFrame | list[dict[str, Any]], file_path:str|Path=None):
    # TODO: use asyncer to save history
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame(history)
    else:
        np.save(file_path, history)


class Register:
    def __init__(self) -> None:
        self.members = {}

    def register(self, name:str) -> None:
        def decorator(func):
            self.members[name] = func
            return func
        return decorator
    
    def __getitem__(self, name:str):
        return self.members[name]
    
    @property
    def all_funcs(self):
        return self.members.values()
    
    @property
    def all_names(self):
        return self.members.keys()


class Recorder():
    def __init__(self, mark:str="J", mode:Literal["max", "min"]="max"):
        self.mark = mark
        self.mode = mode
        self.history = list()

        self._all_columns = set()
    
    def append(self, record:dict):
        assert self.mark in record, \
            f"mark {self.mark} not in record {record}"
        if "_id" not in record:
            record["_id"] = len(self.history)
        self.history.append(record)
        self._all_columns.update(record.keys())
    
    @property
    def dataframe(self):
        return pd.DataFrame(self.history)

    def save_dataframe(self, filename:str|Path, **kwargs):
        df = self.dataframe
        save_history(df, filename)
        return df

    def save_best(self, saved_dir:str|Path, target:str, process_fn=lambda x:x, **kwargs):
        if target not in self.columns:
            target = "_"+target
            if target not in self.columns:
                raise ValueError(f"target {target} not in columns {self.columns}")
        
        if isinstance(saved_dir, str):
            saved_dir = Path(saved_dir)
        saved_dir.mkdir(parents=True, exist_ok=True)

        target_iter, (index, value) = self.get_best_iter()
        target_value = process_fn(target_iter[target])
        if isinstance(target_value, np.ndarray) and target_value.ndim == 1: # 1D array
            save_file = saved_dir / f'{self.mark}-{value:.3f}.csv'
            np.savetxt(save_file, target_value, **kwargs)
        elif isinstance(target_value, np.ndarray) and target_value.ndim == 2: # 2D array
            save_file = saved_dir / f'{self.mark}-{value:.3f}.png'
            plt.imshow(target_value, **kwargs)
            plt.savefig(save_file)
            plt.close()
        else:
            raise ValueError(f"target_value {target_value} has invalid shape {target_value.shape}")
        logger.info(f"{self.mark}@{index}->{value:.3f} saved to {save_file}")
        return target_value, target_iter[self.mark]
    
    def plot(self, target:str, ax=None):
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
        assert index < len(self.history), f"index {index} out of range {len(self.history)}"
        return self.history.iloc[index]
    
    def __add__(self, other:"Recorder"):
        assert self.mark == other.mark and self.target == other.target, "mark and target must be the same"
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
        return {k:v for k,v in info_dict.items() if not k.startswith("_")}

    def get_best_iter(self, mark:str=""):
        mark = mark or self.mark
        res_df = pd.DataFrame(self.history)
        target_id = res_df[mark].argmax() if self.mode == "max" else res_df[mark].argmin()
        return res_df.iloc[target_id], (target_id, res_df.iloc[target_id][mark])

    def get_sublist(self, columns:list[str] | str | None = ""):
        if not columns:
            columns = self.mark
        if isinstance(columns, str):
            return [l.get(columns, np.nan) for l in self.history]
        else:
            return [{k:v for k,v in l.items() if k in columns} for l in self.history]
