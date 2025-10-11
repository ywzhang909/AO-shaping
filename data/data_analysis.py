#%%
import os
import pandas as pd
from dask import dataframe as dd
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchkeras import KerasModel
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import math
from scipy.optimize import curve_fit
# %% U-Net+ Model Definition
class UNetPlus(nn.Module):
    def __init__(self, n_cam_channels=1, f_cam_channels=1, wavefront_channels=1):
        super(UNetPlus, self).__init__()
        
        # Encoder for n_cam (1024, 1280)
        self.n_cam_conv1 = nn.Conv2d(n_cam_channels, 64, kernel_size=3, padding=1)
        self.n_cam_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.n_cam_pool = nn.MaxPool2d(2)
        
        # Encoder for f_cam (192, 192)
        self.f_cam_conv1 = nn.Conv2d(f_cam_channels, 64, kernel_size=3, padding=1)
        self.f_cam_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.f_cam_pool = nn.MaxPool2d(2)
        
        # Fusion layer
        self.fusion_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dec_conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # Output layer to match wavefront size (23, 23)
        self.output_conv = nn.Conv2d(32, wavefront_channels, kernel_size=3, padding=1)
        self.final_upsample = nn.Upsample(size=(23, 23), mode='bilinear', align_corners=False)
        
    def forward(self, cam):
        n_cam, f_cam = cam
        # Process n_cam
        n_cam_enc1 = F.relu(self.n_cam_conv1(n_cam))
        n_cam_enc1 = F.relu(self.n_cam_conv2(n_cam_enc1))
        n_cam_enc2 = self.n_cam_pool(n_cam_enc1)
        n_cam_enc2 = F.relu(nn.Conv2d(64, 128, kernel_size=3, padding=1)(n_cam_enc2))
        n_cam_enc2 = F.relu(nn.Conv2d(128, 128, kernel_size=3, padding=1)(n_cam_enc2))
        
        # Process f_cam
        f_cam_enc1 = F.relu(self.f_cam_conv1(f_cam))
        f_cam_enc1 = F.relu(self.f_cam_conv2(f_cam_enc1))
        f_cam_enc2 = self.f_cam_pool(f_cam_enc1)
        f_cam_enc2 = F.relu(nn.Conv2d(64, 128, kernel_size=3, padding=1)(f_cam_enc2))
        f_cam_enc2 = F.relu(nn.Conv2d(128, 128, kernel_size=3, padding=1)(f_cam_enc2))
        
        # Fusion
        # Resize f_cam_enc2 to match n_cam_enc2 size
        f_cam_enc2_resized = F.interpolate(f_cam_enc2, size=n_cam_enc2.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([n_cam_enc2, f_cam_enc2_resized], dim=1)
        fused = F.relu(self.fusion_conv(fused))
        
        # Decoder
        dec1 = self.upconv1(fused)
        dec1 = F.relu(self.dec_conv1(dec1))
        dec1 = F.relu(self.dec_conv2(dec1))
        
        dec2 = self.upconv2(dec1)
        dec2 = F.relu(self.dec_conv3(dec2))
        dec2 = F.relu(self.dec_conv4(dec2))
        
        # Output
        output = self.output_conv(dec2)
        output = self.final_upsample(output)
         
        return output
# %%
data = pd.read_pickle("data/10/data.pkl")
best_index = data["J"].idxmin()
flatten_v = data.iloc[best_index].v
# %%
class AOShapingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.pkl_path = os.path.join(data_dir, "data.pkl")
        self.img_dir = os.path.join(data_dir, "img")
        self.transform = transform
        self.data = []
        self.load_data()
        
    def load_data(self):
        raw_data = pd.read_pickle(self.pkl_path)
        for index, sample in raw_data.iterrows():
            if index == -1:
                self.data.append((0, f"{index}.npz"))
            else:
                self.data.append((0, f"{index}.npz"))
                self.data.append((1, f"{index}.npz"))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_index, npz_file = self.data[idx]  # 使用列表索引而不是 iloc
        npz_file = np.load(os.path.join(self.img_dir, npz_file))
        f_cam, n_cam, wavefront = npz_file['f_cam'], npz_file['n_cam'], npz_file['wavefront']

        f_cam = f_cam[img_index,:,:]
        n_cam = n_cam[img_index,:,:]
        wavefront = wavefront[img_index,:,:]

        # 转换为 PyTorch 张量
        f_cam = torch.from_numpy(f_cam).float().unsqueeze(0)
        n_cam = torch.from_numpy(n_cam).float().unsqueeze(0)
        wavefront = torch.from_numpy(wavefront).float().unsqueeze(0)
        
        return (n_cam.to("cuda"), f_cam.to("cuda")), wavefront.to("cuda")
     
dataset = AOShapingDataset("data/10")
train_dataset, test_dataset = random_split(dataset, [0.8,0.2])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# %%
net = UNetPlus(n_cam_channels=1, f_cam_channels=1, wavefront_channels=1).to("cuda") 
optimizer = Adam(net.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()
lr_schedule = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
model = KerasModel(net, optimizer=optimizer, loss_fn=loss_fn, lr_scheduler=lr_schedule)
model.fit(train_loader, test_loader, epochs=10, patience=5)

# %%
v = np.load("../last_v.npz")['v'].astype(int)
np.savetxt('to_load-1.csv',v, fmt="%d")
# %%
# 使用HDF5逐步合并所有pkl文件，避免内存溢出
# 创建或打开HDF5文件
dir_name = "4"
with pd.HDFStore(f"{dir_name}/data.h5", mode='w') as hdf_store:
    first_df = True  # 标记是否是第一个DataFrame
    pkl_files = [f for f in os.listdir(dir_name) if f.endswith('.pkl')]
    for file_idx, pkl_file in enumerate(pkl_files):
        
        file_path = os.path.join(dir_name, pkl_file)
        print(f"  Reading {file_path} ({file_idx+1}/{len(pkl_files)})")
        
        # 逐个读取pkl文件
        df = pd.read_pickle(file_path)
        df.columns = [col if not col.startswith("_") else col[1:] for col in df.columns]     
        # 对于第一个DataFrame，直接存储
        if first_df:
            hdf_store.put('data', df, format='table', data_columns=True)
            first_df = False
        else:
            # 对于后续的DataFrame，追加到现有数据中
            hdf_store.append('data', df, format='table', data_columns=True)

# %%
dir = "7"
data_list = [f"{dir}/{i}" for i in os.listdir(dir) if i.endswith(".pkl")]
data_list = [dd.from_pandas(pd.read_pickle(pkl), npartitions=1) for pkl in data_list]
data = dd.concat(data_list, ignore_index=True)
data.columns = [col if not col.startswith("_") else col[1:] for col in data.columns]
data.reset_index(drop=True)

# %%
min_iter = data.loc[data.J.compute().argmin(), :]
max_iter = data.loc[data.J.compute().argmax(), :]
# %%time
df_data = pd.read_pickle('9/11.pkl')
df_data.columns = [col if not col.startswith("_") else col[1:] for col in df_data.columns]
min_iter = df_data.loc[df_data.J.argmin(), :]
max_iter = df_data.loc[df_data.J.argmax(), :]

fig, ax = plt.subplots(3,2)

best_far_spot = min_iter.f_cam[0]
ax[0,0].imshow(best_far_spot)
ax[0,0].set_title(f"far rms={min_iter.J:.4f}")

worst_far_spot = max_iter.f_cam[0]
ax[0,1].imshow(worst_far_spot)
ax[0,1].set_title(f"far rms={max_iter.J:.4f}")

ax[1,0].imshow(min_iter.n_cam[0])
ax[1,1].imshow(max_iter.n_cam[0])

ax[2,0].imshow(min_iter.wavefront[0])
ax[2,1].imshow(max_iter.wavefront[0])

np.savetxt(f"{dir}/min_iter_v.txt", min_iter.v, fmt="%d")

# ax[1].title(f"rms={max_iter.J}") 
# %%
def calculate_ring_energy_radius(image, center=None):
    """
    计算numpy二维图片的环围能量半径
    
    参数:
    image: 二维numpy数组，表示图像
    center: 元组，表示中心点坐标 (y, x)，如果为None则使用图像中心
    
    返回:
    float: 环围能量半径
    """
    # 确保输入是二维数组
    if image.ndim != 2:
        raise ValueError("输入必须是二维数组")
    
    # 获取图像尺寸
    height, width = image.shape
    
    # 如果没有指定中心点，则使用图像中心
    if center is None:
        center_y, center_x = height / 2, width / 2
    else:
        center_y, center_x = center
    
    # 创建坐标网格
    y_indices, x_indices = np.ogrid[:height, :width]
    
    # 计算每个像素到中心点的距离
    distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    
    # 计算总能量
    total_energy = np.sum(image)
    
    # 如果总能量为0，返回0
    if total_energy == 0:
        return 0.0
    
    # 计算加权平均距离（环围能量半径）
    energy_radius = np.sum(distances * image) / total_energy
    
    return energy_radius

def calculate_ring_energy_radius_bins(image, center=None, num_bins=100):
    """
    使用分箱方法计算环围能量半径
    
    参数:
    image: 二维numpy数组，表示图像
    center: 元组，表示中心点坐标 (y, x)，如果为None则使用图像中心
    num_bins: 分箱数量
    
    返回:
    tuple: (半径数组, 能量分布数组, 累积能量分布数组, 能量半径)
    """
    # 确保输入是二维数组
    if image.ndim != 2:
        raise ValueError("输入必须是二维数组")
    
    # 获取图像尺寸
    height, width = image.shape
    
    # 如果没有指定中心点，则使用图像中心
    if center is None:
        center_y, center_x = height / 2, width / 2
    else:
        center_y, center_x = center
    
    # 创建坐标网格
    y_indices, x_indices = np.ogrid[:height, :width]
    
    # 计算每个像素到中心点的距离
    distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    
    # 计算最大距离
    max_distance = np.max(distances)
    
    # 创建距离bins
    bins = np.linspace(0, max_distance, num_bins + 1)
    
    # 计算每个bin中的能量
    energies, _ = np.histogram(distances, bins=bins, weights=image)
    
    # 计算bin中心
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 计算累积能量
    cumulative_energy = np.cumsum(energies)
    
    # 计算总能量
    total_energy = np.sum(energies)
    
    # 如果总能量为0，返回0
    if total_energy == 0:
        return bin_centers, energies, cumulative_energy, 0.0
    
    # 计算加权平均半径
    energy_radius = np.sum(bin_centers * energies) / total_energy
    
    return bin_centers, energies, cumulative_energy, energy_radius


# %%
def find_centroid(img):
    """
    Calculate the centroid coordinates of an object in a binary image.

    Args:
        image (np.ndarray): A binary image represented as a NumPy array.

    Returns:
        tuple: A tuple containing the (y, x) coordinates of the centroid.
    """
    # Calculate centroid
    total = np.sum(img)
    if total == 0:
        raise ValueError("Empty image - all pixel values are zero")
    # Get image dimensions
    height, width = img.shape
    # Create coordinate grids
    x, y = np.indices((width, height))
    # Calculate weighted coordinates
    x_center = int(np.sum(x * img.T) / total)
    y_center = int(np.sum(y * img.T) / total)
    
    return x_center, y_center


def find_lightest_centroid(img):
    """
    Calculate the centroid coordinates of the lightest object in a binary image.

    Args:
        image (np.ndarray): A binary image represented as a NumPy array.

    Returns:
        tuple: A tuple containing the (y, x) coordinates of the centroid.
    """
    # Calculate centroid
    total = np.sum(img)
    if total == 0:
        raise ValueError("Empty image - all pixel values are zero")
    _img = img.copy()
    max_intensity = np.max(_img)
    _img[img!=max_intensity] = 0

    return find_centroid(_img)

def cartesian_to_polar(x, y, c_x, c_y):
    """
    Convert Cartesian coordinates to polar coordinates with a custom origin.

    Args:
        x (np.ndarray): x-coordinates of the points.
        y (np.ndarray): y-coordinates of the points.
        c_x (float): x-coordinate of the custom origin.
        c_y (float): y-coordinate of the custom origin.

    Returns:
        tuple: A tuple containing the radial distances (r) and angles (theta).
    """
    dx = x - c_x
    dy = y - c_y
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    return r, theta

def polar_to_cartesian(r, theta, c_x, c_y):
    """
    Convert polar coordinates to Cartesian coordinates with a custom origin.

    Args:
        r (np.ndarray): Radial distances from the custom origin.
        theta (np.ndarray): Angles in radians.
        c_x (float): x-coordinate of the custom origin.
        c_y (float): y-coordinate of the custom origin.

    Returns:
        tuple: A tuple containing the x-coordinates (x) and y-coordinates (y).
    """
    x = c_x + r * np.cos(theta)
    y = c_y + r * np.sin(theta)
    return x, y

def gaussian(x, A, b, mu, sigma):
    """
    Define the Gaussian function.

    Args:
        x (np.ndarray): Input x values.
        A (float): Amplitude of the Gaussian.
        mu (float): Mean of the Gaussian.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: Output values of the Gaussian function.
    """
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + b

def calculate_diameter(data):
    """
    Calculate the diameter at y = 1/e + b for a given data series.

    Args:
        data (np.ndarray): The data series to fit a Gaussian function.

    Returns:
        float: The calculated diameter, or None if fitting fails or A <= 0.
    """
    x_data = np.arange(len(data))
    initial_guess = [np.max(data), 0, np.argmax(data), 10]
    try:
        popt, _ = curve_fit(gaussian, x_data, data, p0=initial_guess)
        A, b, mu, sigma = popt
        if A > 0:
            diameter = 2 * math.sqrt(2 * sigma**2 * (1 + math.log(A)))
            return diameter
        else:
            print("Amplitude A must be positive for diameter calculation.")
            return None
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None


def calculate_xy_diameters(image, centroid):
    """
    Calculate the diameters at y = 1/e + b in x and y directions.

    Args:
        image (np.ndarray): The input image array.
        centroid (tuple): The (x, y) coordinates of the centroid.

    Returns:
        tuple: A tuple containing the x-direction diameter and y-direction diameter.
    """
    c_x, c_y = centroid
    # Extract data for x and y directions
    y_data = image[:, c_x]
    x_data = image[c_y, :]

    # Calculate diameters
    x_diameter = calculate_diameter(x_data)
    y_diameter = calculate_diameter(y_data)

    return x_diameter, y_diameter

def extract_radial_data(image, centroid_x, centroid_y, angle):
    """
    Extract data along a radial line from the centroid at a given angle.

    Args:
        image (np.ndarray): The input image array.
        centroid_x (int): The x-coordinate of the centroid.
        centroid_y (int): The y-coordinate of the centroid.
        angle (float): The angle in degrees.

    Returns:
        np.ndarray: The extracted data along the radial line.
    """
    height, width = image.shape
    angle_rad = np.deg2rad(angle)
    max_length = int(max(
        math.sqrt(centroid_x**2 + centroid_y**2),
        math.sqrt((width - centroid_x)**2 + centroid_y**2),
        math.sqrt(centroid_x**2 + (height - centroid_y)**2),
        math.sqrt((width - centroid_x)**2 + (height - centroid_y)**2)
    ))
    distances = np.arange(-max_length, max_length + 1)
    x_coords = np.round(centroid_x + distances * np.cos(angle_rad)).astype(int)
    y_coords = np.round(centroid_y + distances * np.sin(angle_rad)).astype(int)
    valid_mask = (0 <= x_coords) & (x_coords < width) & (0 <= y_coords) & (y_coords < height)
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    return image[y_coords, x_coords]


def calculate_diameter_at_angle(image, centroid_x, centroid_y, angle):
    """
    Calculate the diameter at y = 1/e + b at a given angle.

    Args:
        image (np.ndarray): The input image array.
        centroid (tuple): The (y, x) coordinates of the centroid.
        angle (float): The angle in degrees.

    Returns:
        float: The calculated diameter, or None if fitting fails or A <= 0.
    """
    radial_data = extract_radial_data(image, centroid_x, centroid_y, angle)
    return calculate_diameter(radial_data)

# %%
centroid = find_lightest_centroid(best_far_spot)
calculate_xy_diameters(best_far_spot, centroid)
# %%
def get_dia(img: np.ndarray):
    if img.ndim == 3:
        img = img[0]
    centroid = find_lightest_centroid(best_far_spot)
    dia_xy = calculate_xy_diameters(img, centroid)
    
    return centroid, dia_xy

spot_statiscs = data._cam.apply(get_dia)
# %%
spot_statiscs.apply(lambda x: x[1][1]).plot()
# %%
def get_train_data(line:pd.Series):
    y = line['_v']
    x = line['_cam']
    if x.ndim == 3:
        x = x[0]
        
    return x,y
