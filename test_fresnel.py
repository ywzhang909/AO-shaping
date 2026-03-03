"""
测试菲涅尔相位公式与CSV文件的匹配
菲涅尔相位公式: φ(r) = (π / λf) * r²
"""
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
def load_csv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return np.array([[int(x) for x in row[1:]] for row in rows])

csv_300 = load_csv('data/pattern/300mm-focus_10bit.csv')
csv_310 = load_csv('data/pattern/310mm-focus_10bit.csv')

print("=" * 70)
print("菲涅尔相位公式测试")
print("=" * 70)

# 标准菲涅尔相位公式
def fresnel_phase_standard(focal_length, wavelength=532e-9, pixel_size=8e-6):
    """
    标准菲涅尔相位公式 (抛物面近似):
    φ(r) = (π / λf) * r²
    """
    x = np.arange(width) - width // 2
    y = np.arange(height) - height // 2
    X, Y = np.meshgrid(x, y)
    R2 = X**2 + Y**2
    
    # 菲涅尔相位
    phase = (np.pi / wavelength / focal_length) * (R2 * pixel_size**2)
    
    # 包裹到 0~2π
    phase_wrapped = np.mod(phase, 2 * np.pi)
    img = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)
    return img

# 测试不同的波长
def fresnel_phase_custom(focal_length, wavelength, pixel_size=8e-6):
    x = np.arange(width) - width // 2
    y = np.arange(height) - height // 2
    X, Y = np.meshgrid(x, y)
    R2 = X**2 + Y**2
    phase = (np.pi / wavelength / focal_length) * (R2 * pixel_size**2)
    phase_wrapped = np.mod(phase, 2 * np.pi)
    img = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)
    return img

print("\n【测试1】标准菲涅尔公式 (λ=532nm, pixel=8um):")
for fl in [0.3, 0.6, 0.9, 1.2]:
    img = fresnel_phase_standard(fl)
    print(f"  f={fl}m: [0,0]={img[0,0]:4d}, [0,1]={img[0,1]:4d}, [0,2]={img[0,2]:4d}")
print(f"  CSV 300mm:  [0,0]={csv_300[0,0]:4d}, [0,1]={csv_300[0,1]:4d}, [0,2]={csv_300[0,2]:4d}")

print("\n【测试2】搜索匹配300mm CSV的焦距:")
best_f = None
best_diff = 1e15
for fl in np.arange(0.1, 2.0, 0.001):
    img = fresnel_phase_standard(fl)
    diff = np.sum(np.abs(img[0, :20].astype(np.int32) - csv_300[0, :20].astype(np.int32)))
    if diff < best_diff:
        best_diff = diff
        best_f = fl
        if diff < 100:
            print(f"  f={fl:.4f}m: 前20点差异={diff}")
            print(f"    代码: {img[0, :10].tolist()}")
            print(f"    CSV:  {csv_300[0, :10].tolist()}")

print(f"\n  最佳匹配: f={best_f:.4f}m, 差异={best_diff}")

print("\n【测试3】测试不同波长:")
# 也许CSV使用了不同的波长
for wl_nm in [450, 500, 532, 550, 600, 650, 700, 800]:
    wl = wl_nm * 1e-9
    # 反推能匹配[0,0]的焦距
    for fl in np.arange(0.1, 2.0, 0.01):
        img = fresnel_phase_custom(fl, wl)
        if abs(int(img[0,0]) - csv_300[0,0]) < 5:
            print(f"  λ={wl_nm}nm, f={fl:.2f}m: [0,0]={img[0,0]}")
            break

print("\n【测试4】测试不同像素尺寸:")
# 也许CSV使用了不同的像素尺寸
for ps_um in [6, 7, 8, 9, 10, 12, 15, 20]:
    ps = ps_um * 1e-6
    for fl in np.arange(0.1, 2.0, 0.01):
        img = fresnel_phase_custom(fl, wavelength, ps)
        if abs(int(img[0,0]) - csv_300[0,0]) < 5:
            print(f"  pixel={ps_um}um, f={fl:.2f}m: [0,0]={img[0,0]}")
            break

print("\n【测试5】反推CSV的实际参数:")
# 从CSV数据反推 (π / λf) 的值
# 相位差 = (π / λf) * pixel_size² * (r2² - r1²)
# 对于x=0和x=1（第一行，y=-600）
x0, x1 = 0, 1
y = 0  # 第一行
center_x, center_y = width // 2, height // 2

r0_sq = (x0 - center_x)**2 + (y - center_y)**2
r1_sq = (x1 - center_x)**2 + (y - center_y)**2
delta_r2 = r1_sq - r0_sq

val0, val1 = csv_300[0, 0], csv_300[0, 1]
# 考虑相位包裹
delta_val = val1 - val0  # 745 - 549 = 196
# 实际相位差（考虑2π包裹）
# 如果196 < 512, 假设没有包裹
delta_phase = (delta_val / 1023) * 2 * np.pi

# k = π / (λf) = delta_phase / (pixel_size² * delta_r2)
k = delta_phase / (pixel_size**2 * delta_r2)
f_estimated = np.pi / (wavelength * k)

print(f"  从CSV第一行反推:")
print(f"    r0² = {r0_sq}, r1² = {r1_sq}")
print(f"    Δr² = {delta_r2}")
print(f"    灰度差 = {delta_val}")
print(f"    相位差 = {delta_phase:.4f} rad")
print(f"    k = π/(λf) = {k:.2f}")
print(f"    估计焦距 f = {f_estimated:.4f} m = {f_estimated*1000:.1f} mm")

# 使用估计的焦距生成图像
img_estimated = fresnel_phase_standard(f_estimated)
print(f"\n  使用 f={f_estimated:.4f}m:")
print(f"    代码: {img_estimated[0, :10].tolist()}")
print(f"    CSV:  {csv_300[0, :10].tolist()}")

# 验证
period_csv = 5  # 从前面的分析，CSV周期约为5像素
period_est = None
if f_estimated > 0:
    diffs = np.diff(img_estimated[0, :50])
    wraps = np.where(diffs < -500)[0]
    if len(wraps) >= 2:
        period_est = wraps[1] - wraps[0]
print(f"\n  周期对比: CSV={period_csv}像素, 估计={period_est}像素")

print("\n" + "=" * 70)
print("结论:")
if f_estimated > 0 and f_estimated < 1.0:
    print(f"CSV文件的实际焦距约为 {f_estimated*1000:.0f}mm，不是标称的300mm")
    print(f"建议修改代码中的焦距参数为 {f_estimated:.4f} 来匹配CSV")
else:
    print("无法通过标准菲涅尔公式匹配CSV文件")
