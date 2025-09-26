#%%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms.v2 import ToTensor, Compose, ToDtype, Resize
from timm.models import create_model
from torchkeras import KerasModel
import matplotlib.pyplot as plt

def resave_pickle(file_path, compression="zip"):
    data = pd.read_pickle(file_path, compression=compression)
    new_columns = []
    for col in data.columns:
        if col.startswith("_"):
            clean_col = col[1:]
            if clean_col in data.columns:
                merged_data = []
                for i in range(len(data)):
                    clean_data = data[clean_col].iloc[i]
                    merged_data =  data[col].iloc[i] if pd.isna(clean_data) else clean_data
                data[clean_col] = merged_data
                data.drop(columns=[col], inplace=True)
            else:
                new_columns.append(clean_col)
        else:
            new_columns.append(col)
            
    data.columns = new_columns
    data.to_pickle(file_path, compression=compression)
# %%
data1 = pd.read_pickle("7.pkl", compression="zip")
data2 = pd.read_pickle("8.pkl", compression="zip")

data = pd.concat([data1, data2], axis=0)
best_index = data["J"].idxmin()
flatten_v = data.iloc[best_index].v
# %%
class AOShapingDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        img = sample.cam
        if img.ndim == 2:
            img = np.stack([img,img], axis=0) # shape(2, 144, 144)
        img = img.transpose([1,2,0])
        v = flatten_v-sample.v
        v = torch.from_numpy(v[1:]).float()
        if self.transform:
            img = self.transform(img)
        return img, v
    
transforms = Compose([
    ToTensor(),
    ToDtype(torch.float32, scale=True),
    Resize((192,192))
])

dataset = AOShapingDataset(data, transform=transforms)
train_dataset, test_dataset = random_split(dataset, [0.8,0.2])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# %%
net = create_model("vgg13", in_chans=2, num_classes=64, pretrained=False)
optimizer = Adam(net.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()
lr_schedule = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
model = KerasModel(net, optimizer=optimizer, loss_fn=loss_fn, lr_scheduler=lr_schedule)
model.fit(train_loader, test_loader, epochs=10, patience=5)
# %%

def visualize_conv_kernels(model, layer_name="features"):
    """
    可视化VGG模型的卷积核
    
    Args:
        model: 训练好的VGG模型
        layer_name: 包含卷积层的模块名称，默认为"features"
    """
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append((name, module))
    
    if not conv_layers:
        print("No convolutional layers found in the model.")
        return
    
    # 计算总的子图数量
    total_subplots = sum(min(layer.weight.shape[0], 8) * min(layer.weight.shape[1], 8) 
                         for _, layer in conv_layers)
    
    # 创建足够大的图形
    fig, axes = plt.subplots(nrows=len(conv_layers), ncols=8, figsize=(20, 2*len(conv_layers)))
    if len(conv_layers) == 1:
        axes = [axes]
    
    for i, (name, layer) in enumerate(conv_layers):
        weights = layer.weight.data.cpu().numpy()
        
        # 显示前8个卷积核（如果有的话）
        for j in range(min(weights.shape[0], 8)):
            ax = axes[i][j] if len(conv_layers) > 1 else axes[j]
            # 显示第一个输入通道的权重
            kernel = weights[j, 0, :, :] if weights.shape[1] > 0 else weights[j, 0, :, :]
            im = ax.imshow(kernel, cmap='viridis')
            ax.set_title(f'{name}\nKernel {j}')
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig("conv_kernels.png")
    plt.show()

def visualize_feature_maps(model, input_tensor, layer_names=["features.0", "features.2", "features.5", "features.7"]):
    """
    可视化输入数据经过VGG模型各层后的特征图
    
    Args:
        model: 训练好的VGG模型
        input_tensor: 输入数据张量
        layer_names: 要可视化的层名称列表
    """
    activations = {}
    
    def hook_fn(module, input, output, name):
        activations[name] = input[0].detach().cpu().numpy()
    
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hook = module.register_forward_hook(lambda module, input, output, n=name: hook_fn(module, input, output, n))
            hooks.append(hook)
    
    # 前向传播
    with torch.no_grad():
        _ = model(input_tensor.unsqueeze(0))
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 可视化特征图
    for layer_name in layer_names:
        if layer_name in activations:
            activation = activations[layer_name]
            num_features = min(activation.shape[1], 8)  # 最多显示8个特征图
            
            if num_features <= 0:
                continue
                
            fig, axes = plt.subplots(nrows=1, ncols=num_features, figsize=(15, 3))
            if num_features == 1:
                axes = [axes]
            for i in range(num_features):
                ax = axes[i]
                feature_map = activation[0, i, :, :]
                im = ax.imshow(feature_map, cmap='viridis')
                ax.set_title(f'{layer_name}\nFeature Map {i}')
                ax.axis('off')
                
            plt.suptitle(f"Feature Maps for Layer: {layer_name}")
            plt.tight_layout()
            plt.savefig(f"feature_maps_{layer_name.replace('.', '_')}.png")
            plt.show()

# 可视化VGG卷积核
visualize_conv_kernels(net)

# 准备data1和data2的第一行数据
sample1_cam = data1.iloc[0].cam
sample2_cam = data2.iloc[0].cam

# 处理数据以匹配模型输入要求
def prepare_image(img):
    if img.ndim == 2:
        img = np.stack([img, img], axis=0)  # shape(2, 144, 144)
    img = img.transpose([1, 2, 0])  # 转换为(H, W, C)
    img_tensor = transforms(img)  # 应用变换
    return img_tensor

sample1_tensor = prepare_image(sample1_cam)
sample2_tensor = prepare_image(sample2_cam)

# 可视化data1.iloc[0].cam经过卷积核之后的图片
print("Visualizing feature maps for data1.iloc[0].cam")
visualize_feature_maps(net, sample1_tensor)

# 可视化data2.iloc[0].cam经过卷积核之后的图片
print("Visualizing feature maps for data2.iloc[0].cam")
visualize_feature_maps(net, sample2_tensor)
