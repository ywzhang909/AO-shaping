#%%
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms.v2 import ToTensor, Compose, ToDtype, Resize
from timm.models import create_model
from torchkeras import KerasModel
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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