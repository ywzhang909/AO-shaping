import numpy as np
import numba
import cupy as cp

from scipy.ndimage import center_of_mass

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from typing import Tuple

@numba.njit
def calculate_sharpness_numba(img:np.ndarray):
    h, w = img.shape
    gradient_x = np.zeros_like(img)
    gradient_y = np.zeros_like(img)
    for i in range(h):
        for j in range(1, w-1):
            gradient_x[i, j] = (img[i, j+1] - img[i, j-1]) / 2
        gradient_x[i, 0] = img[i, 1] - img[i, 0]
        gradient_x[i, w-1] = img[i, w-1] - img[i, w-2]
    for i in range(1, h-1):
        for j in range(w):
            gradient_y[i, j] = (img[i+1, j] - img[i-1, j]) / 2
    for j in range(w):
        gradient_y[0, j] = img[1, j] - img[0, j]
        gradient_y[h-1, j] = img[h-1, j] - img[h-2, j]
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    sharpness = np.mean(gradient_magnitude)
    return sharpness

def calculate_sharpness_cupy(img:cp.ndarray):
    gradient_x = cp.gradient(img, axis=1)
    gradient_y = cp.gradient(img, axis=0)
    gradient_magnitude = cp.sqrt(gradient_x**2 + gradient_y**2)
    sharpness = cp.mean(gradient_magnitude)
    return sharpness

def calculate_sharpness(img:np.ndarray):
  gradient_x = np.gradient(img, axis=1)
  gradient_y = np.gradient(img, axis=0)
  gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
  sharpness = np.mean(gradient_magnitude)
  return sharpness

@numba.njit
def crop_numba(img:np.ndarray, sample_pix=500):
    assert img.ndim == 2
    h, w = img.shape
    bg = 0.0
    for i in range(min(sample_pix, h)):
        for j in range(w):
            if img[i, j] > bg:
                bg = img[i, j]
    rows = np.zeros(h, dtype=np.bool_)
    for i in range(h):
        for j in range(w):
            if img[i, j] > bg:
                rows[i] = True
                break
    cols = np.zeros(w, dtype=np.bool_)
    for j in range(w):
        for i in range(h):
            if img[i, j] > bg:
                cols[j] = True
                break
    rmin = h
    rmax = -1
    for i in range(h):
        if rows[i]:
            if i < rmin:
                rmin = i
            if i > rmax:
                rmax = i
    cmin = w
    cmax = -1
    for j in range(w):
        if cols[j]:
            if j < cmin:
                cmin = j
            if j > cmax:
                cmax = j
    if rmin > rmax or cmin > cmax:
        return img[0:0, 0:0]  # empty
    return img[rmin:rmax + 1, cmin:cmax + 1]

def crop_cupy(img:cp.ndarray, sample_pix=500):
    assert img.ndim == 2
    bg = cp.max(img[:sample_pix,:])
    rows = cp.any(img>bg, axis=1)
    cols = cp.any(img>bg, axis=0)
    rmin, rmax = cp.nonzero(rows)[0][[0, -1]]
    cmin, cmax = cp.nonzero(cols)[0][[0, -1]]

    return img[rmin:rmax + 1, cmin:cmax + 1]

def crop(img:np.ndarray, sample_pix=500):
  assert img.ndim == 2
  bg = np.max(img[:sample_pix,:])
  rows = np.any(img>bg, axis=1)
  cols = np.any(img>bg, axis=0)
  rmin, rmax = np.nonzero(rows)[0][[0, -1]]
  cmin, cmax = np.nonzero(cols)[0][[0, -1]]

  return img[rmin:rmax + 1, cmin:cmax + 1]

def center_of_mass_cupy(intensity:cp.ndarray, xv:cp.ndarray, yv:cp.ndarray, moment:int=1) -> Tuple[float, float]:
    _intensity = intensity.copy().astype(cp.float32)**moment
    total_intensity = cp.sum(_intensity)
    c_x = cp.sum(xv * _intensity) / total_intensity
    c_y = cp.sum(yv * _intensity) / total_intensity
    return (float(c_x), float(c_y))

@numba.njit
def center_of_mass_numba(intensity:np.ndarray, xv:np.ndarray, yv:np.ndarray, moment:int=1) -> Tuple[float, float]:
    """
    计算光强的中心位置

    :param intensity: 强度分布
    :param x: x坐标矩阵
    :param y: y坐标矩阵
    :param moment: 中心位置的阶数
    :return center_x, center_y: 光强的中心位置
    """
    _intensity = intensity.copy().astype(np.float32)**moment
    total_intensity = np.sum(_intensity)
    c_x = np.sum(xv * _intensity) / total_intensity
    c_y = np.sum(yv * _intensity) / total_intensity
    return (float(c_x), float(c_y))

def center_of_brightness_cupy(img:cp.ndarray) -> Tuple[int, int]:
    center = cp.unravel_index(cp.argmax(img), img.shape)[::-1]
    return int(center[0]), int(center[1])

def center_of_brightness_numba(img:np.ndarray) -> Tuple[int, int]:
    center = np.unravel_index(np.argmax(img), img.shape)[::-1]
    return int(center[0]), int(center[1])

def diffraction_limit(lamd, aperture, dist):
    """
    衍射极限光斑直径

    :param lamd: 波长
    :param aperture: 发射光孔径
    :param dist: 传输距离
    :return: 衍射极限光斑直径
    """

    return 1.22 * lamd * dist / aperture

def jitter_diameter(lamd, aperture, dist):
    """
    光束抖动直径，用于自适应光学

    :param lamd: 波长
    :param aperture: 发射光孔径
    :param dist: 传输距离
    :return diameter: 光束抖动直径
    """
    if dist < 1e4:
        diameter = 1e-6 * dist
    else:
        diameter = 3 * lamd * dist / aperture

    return diameter

def centroid(intensity:np.ndarray, moment:int=1, threshold=0.01) -> Tuple[int, int]:
    """
    光强的质心位置

    :param intensity: 强度分布
    :param x: x坐标矩阵
    :param y: y坐标矩阵
    :return center_x, center_y: 光强的质心
    """
    _intensity = intensity.copy().astype(np.float32)
    _intensity -= threshold*np.max(_intensity)
    _intensity[_intensity < 0] = 0
    
    center = center_of_mass(_intensity**moment)
    return int(center[1]), int(center[0])

def peak_position(intensity, x, y):
    """
    光强峰值处的坐标

    :param intensity: 强度分布
    :param x: x坐标矩阵
    :param y: y坐标矩阵
    :return xp, yp: 光强峰值处的坐标
    """
    index = np.unravel_index(intensity.argmax(), intensity.shape)
    xp, yp = x[index[0], index[1]], y[index[0], index[1]]

    return xp, yp

def make_coord(img:np.ndarray):
    """
    生成坐标矩阵

    :param img: 强度分布
    :return x, y: 坐标矩阵
    """
    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    return x, y

def radius(intensity, center='centroid', energy=0.99):
    """
    以center为圆心，占总能量百分比为energy的圆的半径

    :param intensity: 强度分布
    :param x: x坐标矩阵
    :param y: y坐标矩阵
    :param center: 圆心，默认'centroid'，可选'peak'，'origin'，坐标，如(0, 0)
    :param energy: 圆内的能量比，默认0.99，取值范围0~1，常用0.5，0.865， 0.99
    :return radius: 圆的半径
    """
    x, y = make_coord(intensity)
    npix = len(x)
    dpix = x[0, 1] - x[0, 0]

    if type(center) is str:
        if center == 'peak':
            x0, y0 = peak_position(intensity, x, y)
        elif center == 'centroid':
            x0, y0 = centroid(intensity)
        elif center == 'origin':
            h, w = intensity.shape
            x0, y0 = w // 2, h // 2
        else:
            raise ValueError('center is wrong set')
    else:
        x0, y0 = center[0], center[1]

    power_in_circle = np.sum(intensity) * energy
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    radius = npix * dpix / 2
    radius_change = npix * dpix / 4

    for i in range(300):
        mask = (np.sign(radius - r) + 1) / 2
        power = np.sum(intensity * mask)

        if power - power_in_circle < -1e-10 * power_in_circle:
            radius += radius_change
        elif power - power_in_circle > 1e-10 * power_in_circle:
            radius -= radius_change
        else:
            break

        radius_change /= 2
        if radius_change < dpix / 50:
            break
        if i == 299:
            raise StopIteration('Maximal number of iterations reached while calculating beam radius.')

    return radius

def effective_radius(intensity, dpix, clip):
    """
    有效光斑半径

    :param intensity: 强度分布
    :param dpix: 网格尺寸
    :param clip: 阈值，光强大于峰值*clip的像素计入有效光斑面积
    :return radius: 光斑有效半径
    """
    threshold = intensity.max() * clip
    above = (np.sign((intensity - threshold)) + 1) / 2
    d_effective = (above.sum() * dpix ** 2 * 4 / np.pi) ** 0.5

    return d_effective / 2

def power_bucket(intensity, x, y, center, r_bucket, weighted=4):
    """
    以center为圆心，r_bucket为半径的桶中功率

    :param intensity: 强度分布
    :param x: x坐标矩阵
    :param y: y坐标矩阵
    :param center: 桶的圆心，默认'centroid'，可选'peak'，'origin'，坐标，如(0, 0)
    :param r_bucket: 桶半径
    :return: 桶中功率
    """
    dpix = x[0, 1] - x[0, 0]
    radius = r_bucket

    if type(center) is str:
        if center == 'peak':
            x0, y0 = peak_position(intensity, x, y)
        elif center == 'centroid':
            x0, y0 = centroid(intensity)
        elif center == 'origin':
            x0, y0 = 0, 0
        else:
            raise ValueError('center is wrong set')
    else:
        x0, y0 = center[0], center[1]

    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    _weight = weighted  if weighted > 0 else 1
    weights = np.exp(-(r**_weight) / (2 * (r_bucket/2)**2))
    mask = (np.sign(radius - r) + 1) / 2
    intensity_in_bucket = intensity * mask * weights
    power_in_bucket = intensity_in_bucket.sum() * dpix ** 2

    return power_in_bucket

def disp(img, xv, yv, r_bucket, threshold=0, title=''):
    center = centroid(img, threshold=threshold)
    
    def calc_j():
            r = int(r_bucket)
            _img = img.copy()
            _img[img<5] = 0
            in_power = np.sum(_img[center[1]-r:center[1]+r, center[0]-r:center[0]+r])          
            return in_power
    
    _, ax = plt.subplots()
    plt.imshow(img)
    ax.scatter(center[0], center[1], c='r', marker='x')
    ax.add_patch(Rectangle((center[0]-r_bucket, center[1]-r_bucket), 2*r_bucket, 2*r_bucket, edgecolor='red', facecolor='none'))
    J = calc_j()
    plt.title(f'Image {title}. {J=}')
    plt.colorbar()
    plt.show()