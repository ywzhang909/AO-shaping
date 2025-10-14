#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.transforms.v2 import Compose, ToTensor, Resize
from torch.utils.data import Dataset,DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchkeras import KerasModel
import segmentation_models_pytorch as smp

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
IMG_SIZE = (256, 256)
model = smp.Unet(
    encoder_name="resnet34",        # 编码器名称
    in_channels=2,                  # 输入通道数为 2（n_cam 和 f_cam）
    classes=0,                      # 输出通道数为 1（波前）
    encoder_weights=None
)
print(model)

flatten_v = np.loadtxt(os.path.join(DATA_DIR, "flatten_voltages.csv"))
plt.bar(np.arange(len(flatten_v)), flatten_v)
plt.xlabel("Voltage Index")
plt.ylabel("Voltage Value")
plt.title("Flatten Voltages")
plt.show()

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
        # TODO wavefront归一化（0-1）；nan处理成特殊值0？
        wavefront = (wavefront - np.nanmin(wavefront)) / (np.nanmax(wavefront) - np.nanmin(wavefront))
        wavefront = np.nan_to_num(wavefront, nan=0.0)

        # 转换为 PyTorch 张量
        f_cam_t = self.transform(f_cam)
        n_cam_t = self.transform(n_cam)
        wavefront_t = self.transform(wavefront)
        
        return torch.concat([n_cam_t, f_cam_t], dim=0), wavefront_t
     
transform = Compose([
    ToTensor(),
    Resize(IMG_SIZE),
])
dataset = AOShapingDataset(os.path.join(DATA_DIR, "10"), transform=transform)
train_dataset, test_dataset = random_split(dataset, [0.8,0.2])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %%
# get one data sample and display
cam, wavefront = dataset[0]
n_cam, f_cam = cam[0,:,:], cam[1,:,:]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(n_cam.squeeze(), cmap='gray')
plt.title('n_cam')
plt.subplot(1, 3, 2)
plt.imshow(f_cam.squeeze(), cmap='gray')
plt.title('f_cam')
plt.subplot(1, 3, 3)
plt.imshow(wavefront.squeeze(), cmap='gray')
plt.title('wavefront')
plt.show()

print(np.nanmax(wavefront), np.nanmin(wavefront))
# %%
optimizer = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.L1Loss()
lr_schedule = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

model = KerasModel(model, optimizer=optimizer, loss_fn=loss_fn, lr_scheduler=lr_schedule)
model.fit(train_loader, test_loader, epochs=10, patience=5)
# %%
