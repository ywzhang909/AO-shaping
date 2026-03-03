import numpy as np
import csv

# SLM 参数
RESOLUTION = (1920, 1200)
Bits = 10
wavelength = 532e-9
pixel_size = 8e-6

height, width = RESOLUTION[1], RESOLUTION[0]
max_val = 2 ** Bits - 1

# 加载CSV文件
with open('data/pattern/300mm-focus_10bit.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

csv_array = np.array([[int(x) for x in row[1:]] for row in rows])

print("=== 查找匹配CSV的焦距 ===")

# 代码生成函数
def generate_focus(focal_length, wavelength=532e-9, pixel_size=8e-6):
    x = np.arange(width) - width // 2
    y = np.arange(height) - height // 2
    X, Y = np.meshgrid(x, y)
    R2 = X**2 + Y**2
    phase = (np.pi / wavelength / focal_length) * (R2 * pixel_size**2)
    phase_wrapped = np.mod(phase, 2 * np.pi)
    return (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)

# 手动检查不同焦距下的第一行模式
print(f"CSV第一行前15: {csv_array[0, :15].tolist()}")

print(f"\n测试不同焦距:")
for fl in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
    img = generate_focus(fl)
    print(f"f={fl:.1f}m: [0,0]={img[0,0]:4d}, [0,1]={img[0,1]:4d}, [0,2]={img[0,2]:4d}, [0,3]={img[0,3]:4d}, [0,4]={img[0,4]:4d}")

# 搜索最佳匹配
print(f"\n搜索最佳匹配...")
best_f = None
best_diff = 1e15
for fl in np.arange(0.1, 3.0, 0.001):
    img = generate_focus(fl)
    # 只比较前几个点
    diff = np.sum(np.abs(img[0, :15].astype(np.int32) - csv_array[0, :15].astype(np.int32)))
    if diff < best_diff:
        best_diff = diff
        best_f = fl
        if diff < 50:
            print(f"f={fl:.4f}m: 差异={diff}, [0,:15]={img[0,:15].tolist()}")

print(f"\n最佳匹配: f={best_f:.4f}m, 前15点差异={best_diff}")

# 使用最佳焦距生成完整图像并对比
if best_f:
    img = generate_focus(best_f)
    print(f"\n完整对比 (f={best_f:.4f}m):")
    print(f"代码 [0,:15]: {img[0, :15].tolist()}")
    print(f"CSV  [0,:15]: {csv_array[0, :15].tolist()}")
    print(f"代码 [1,:15]: {img[1, :15].tolist()}")
    print(f"CSV  [1,:15]: {csv_array[1, :15].tolist()}")
    
    total_diff = np.sum(np.abs(img.astype(np.int32) - csv_array.astype(np.int32)))
    print(f"\n全图总差异: {total_diff}")

# 检查310mm
print(f"\n=== 检查310mm ===")
with open('data/pattern/310mm-focus_10bit.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows310 = list(reader)
csv310 = np.array([[int(x) for x in row[1:]] for row in rows310])

print(f"310mm CSV [0,:15]: {csv310[0, :15].tolist()}")

best_f_310 = None
best_diff_310 = 1e15
for fl in np.arange(0.1, 3.0, 0.001):
    img = generate_focus(fl)
    diff = np.sum(np.abs(img[0, :15].astype(np.int32) - csv310[0, :15].astype(np.int32)))
    if diff < best_diff_310:
        best_diff_310 = diff
        best_f_310 = fl
        if diff < 50:
            print(f"310mm: f={fl:.4f}m: 差异={diff}")

print(f"310mm 最佳匹配: f={best_f_310:.4f}m, 差异={best_diff_310}")
if best_f_310:
    img_310 = generate_focus(best_f_310)
    print(f"代码 [0,:15]: {img_310[0, :15].tolist()}")
    print(f"CSV  [0,:15]: {csv310[0, :15].tolist()}")

# 验证比例
if best_f and best_f_310:
    print(f"\n=== 验证 ===")
    print(f"300mm 实际焦距: {best_f*1000:.2f} mm")
    print(f"310mm 实际焦距: {best_f_310*1000:.2f} mm")
    print(f"实际焦距比: {best_f_310/best_f:.4f}")
    print(f"名义比 310/300: {310/300:.4f}")
