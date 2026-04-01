# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from ao_shaping.algorithm.target_func import ImageTargetFunc
plt.gray()
# %%
data = pd.read_pickle('../data/wf-less/20260331_170725/cd3b8e90-bf3d-45a2-b8fd-a9b5556dff7f.pkl', compression='zip')

init_data = data.iloc[0]
process_data = data.iloc[1:]
data.pib.plot()
# %%
img = process_data.iloc[0]._img
h, w = img.shape
center = w//2, h//2

func = ImageTargetFunc(w, h,(w//2, h//2))
process_data._img.apply(func.fit_gaussian_radius).plot()
# %%
process_data._img.apply(func.second_moment_radius).plot()
# %%
process_data.pib.plot()
# %%
cx, cy = func.center_of_mass(img)
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.scatter(cx, cy, c='red', s=100, marker='x', label='Center of Mass')

# 计算两个半径
radius_sm = func.second_moment_radius(img, center)
radius_gauss = func.fit_gaussian_radius(img, center)
if radius_gauss is None:
    radius_gauss = 0

# 分别绘制两个圆：以画面中心为圆心
circle_sm = Circle((w//2, h//2), radius_sm, fill=False, color='yellow', linewidth=2, label=f'Second Moment Radius: {radius_sm:.1f}')
circle_gauss = Circle((w//2, h//2), radius_gauss, fill=False, color='cyan', linewidth=2, linestyle='--', label=f'Gaussian Radius: {radius_gauss:.1f}')
plt.gca().add_patch(circle_sm)
plt.gca().add_patch(circle_gauss)
plt.colorbar()
plt.legend()
plt.title(f'Second Moment Radius: {radius_sm:.1f}, Gaussian Radius: {radius_gauss:.1f}')
plt.show()
# %%
from matplotlib.animation import FuncAnimation

# 创建动画展示process_data._img变化过程
fig, ax = plt.subplots(figsize=(8, 6))

def init():
    ax.clear()
    return []

def update(frame):
    ax.clear()
    img = process_data.iloc[frame]._img
    ax.imshow(img)
    ax.set_title(f'Frame {frame}: PIB = {process_data.iloc[frame].pib:.4f}')
    return []

anim = FuncAnimation(fig, update, frames=len(process_data), init_func=init, interval=100, blit=True)
# 保存为gif
anim.save('pib_process.gif', writer='pillow', fps=10)
# %%
