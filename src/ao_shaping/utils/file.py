import os
import re

import uuid
from pathlib import Path
from glob import glob
from datetime import datetime

import numpy as np

def gen_file_name_inc(dir: str|Path, postfix: str = ''):
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
    path = dir.joinpath(fname).with_suffix(postfix)
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
        return None
        
    min_rms = min([get_rms(f) for f in glob(f"{data_path}/rms-*.csv") if get_rms(f) is not None])
    try:
        init_V = np.loadtxt(f"{data_path}/rms-{min_rms:.3f}.csv")
        print(f"init_V by rms {min_rms:.3f}")
    except FileNotFoundError:
        init_V = np.zeros(64)
        print(f"init_V by rms {min_rms:.3f} not found, return 0")
    return init_V

def get_init_V_by_energy(date:str = ''):
    data_path = f"data/flatten_voltages/{date}" if date else f"data/flatten_voltages/{datetime.now().strftime('%Y%m%d')}"
    def get_energy(file_name):
        regex_pattern = r"to_load_V-(\d+\.\d+)\.csv"
        match = re.search(regex_pattern, file_name)
        if match:
            return float(match.group(1))
        return None
    max_energy = max([get_energy(f) for f in glob(f"{data_path}/to_load_V-*.csv") if get_energy(f) is not None])
    try:
        print(f"init_V by energy {max_energy:.3f}")
        init_V = np.loadtxt(f"{data_path}/to_load_V-{max_energy:.3f}.csv")
    except FileNotFoundError:
        init_V = np.zeros(64)
        print(f"init_V by energy {max_energy:.3f} not found, return 0")
    return init_V