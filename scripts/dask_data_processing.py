#%%
import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.v2 import ToTensor, Compose, ToDtype, Resize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def read_pkl_files_with_dask(directory="data/10", npartitions=4):
    """
    使用 dask 读取指定目录中的所有 .pkl 文件并合并为一个 dask DataFrame
    """
    # 获取所有 .pkl 文件路径
    pkl_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pkl')]
    
    # 读取所有 .pkl 文件并创建 dask DataFrame 列表
    ddf_list = []
    for file_path in pkl_files:
        # 读取单个文件为 pandas DataFrame
        df = pd.read_pickle(file_path)
        df.columns = [c.replace('_','') for c in df.columns]
        # 转换为 dask DataFrame
        ddf = dd.from_pandas(df, npartitions=npartitions)
        ddf_list.append(ddf)
    
    # 合并所有 dask DataFrame
    if len(ddf_list) > 1:
        combined_ddf = dd.concat(ddf_list, interleave_partitions=True)
    elif len(ddf_list) == 1:
        combined_ddf = ddf_list[0]
    else:
        raise ValueError("No .pkl files found in the directory")
    
    return combined_ddf

class DaskAOShapingDataset(Dataset):
    """
    兼容 dask 的 AOShaping 数据集类
    """
    def __init__(self, dask_data, flatten_v, transform=None):
        self.dask_data = dask_data
        self.flatten_v = flatten_v
        self.transform = transform
        # 计算数据集大小（需要使用 compute()）
        self.length = len(dask_data)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 从 dask DataFrame 获取单行数据（需要使用 compute()）
        sample = self.dask_data.iloc[idx:idx+1].compute().iloc[0]
        img = sample.cam
        if img.ndim == 2:
            img = np.stack([img, img], axis=0)  # shape(2, 144, 144)
        img = img.transpose([1, 2, 0])
        v = self.flatten_v - sample.v
        v = torch.from_numpy(v[1:]).float()
        if self.transform:
            img = self.transform(img)
        return img, v

# 定义数据变换
transforms = Compose([
    ToTensor(),
    ToDtype(torch.float32, scale=True),
    Resize((192, 192))
])

def process_data_with_dask():
    """
    使用 dask 处理数据的主要函数
    """
    # 启动 dask 客户端（可选，用于监控）
    # client = Client(processes=False)  # 使用线程而非进程以减少内存开销
    # print("Dask dashboard link:", client.dashboard_link)
    
    # 读取并合并所有 .pkl 文件
    data = read_pkl_files_with_dask()
    
    # 计算最佳索引（J 值最小的行）
    best_index = data["J"].idxmin().compute()
    
    # 获取最佳行的 v 值
    # 注意：这里需要先计算 best_index，然后获取对应的行
    # 由于 dask 的惰性计算特性，我们需要使用 compute()
    flatten_v = data.loc[best_index, "v"].compute()
    
    return data, flatten_v, best_index

def create_dask_data_loaders(batch_size=32, train_split=0.8):
    """
    使用 dask 创建数据加载器
    """
    # 处理数据
    data, flatten_v, best_index = process_data_with_dask()
    
    # 创建数据集
    dataset = DaskAOShapingDataset(data, flatten_v, transform=transforms)
    
    # 划分训练集和测试集
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, data, flatten_v, best_index

def visualize_best_v(best_v):
    """
    可视化 best_v 数据
    修复了 ValueError: could not convert string to float 的问题
    """
    # 检查 best_v 的类型并进行适当处理
    if isinstance(best_v, str):
        # 如果 best_v 是字符串，尝试解析它
        # 移除方括号和多余的空格
        cleaned_str = best_v.strip('[] \n')
        # 分割字符串并转换为浮点数数组
        try:
            best_v_array = np.array([float(x) for x in cleaned_str.split() if x])
        except ValueError:
            # 如果解析失败，尝试其他方法
            print("无法直接解析字符串为浮点数数组，尝试使用 eval")
            try:
                best_v_array = np.array(eval(best_v))
            except:
                print("无法解析 best_v 数据")
                return
    elif isinstance(best_v, (list, np.ndarray)):
        # 如果 best_v 已经是数组，直接转换为 numpy 数组
        best_v_array = np.array(best_v)
    else:
        # 其他情况，尝试直接转换
        try:
            best_v_array = np.array(best_v)
        except:
            print(f"无法处理 best_v 数据，类型为: {type(best_v)}")
            return
    
    # 绘制图形
    plt.figure(figsize=(12, 6))
    plt.plot(best_v_array)
    plt.title("Best V Values")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

# 如果直接运行此脚本，则执行数据处理
if __name__ == "__main__":
    print("开始使用 dask 处理数据...")
    data, flatten_v, best_index = process_data_with_dask()
    print(f"数据处理完成。最佳索引: {best_index}")
    print(f"数据形状: {data.shape[0].compute()} 行")
    print("前5行 J 值:")
    print(data["J"].head().compute())
    
    # 可视化 J 值
    plt.figure(figsize=(12, 6))
    plt.plot(data["J"].compute())
    plt.title("J Values")
    plt.xlabel("Index")
    plt.ylabel("J Value")
    plt.grid(True)
    plt.show()
    
    # 获取最佳行的 v 值并可视化
    best_v = data.loc[best_index, "v"].compute()
    print(f"Best V: {best_v}")
    visualize_best_v(best_v)
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, test_loader, data, flatten_v, best_index = create_dask_data_loaders()
    print(f"训练集批次数量: {len(train_loader)}")
    print(f"测试集批次数量: {len(test_loader)}")
