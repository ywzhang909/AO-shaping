import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from typing import Tuple

def calculate_sharpness(img:np.ndarray):
  gradient_x = np.gradient(img, axis=1)
  gradient_y = np.gradient(img, axis=0)
  gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
  sharpness = np.mean(gradient_magnitude)
  return sharpness

def crop(img:np.ndarray, sample_pix=500):
  assert img.ndim == 2
  bg = np.max(img[:sample_pix,:])
  rows = np.any(img>bg, axis=1)
  cols = np.any(img>bg, axis=0)
  rmin, rmax = np.nonzero(rows)[0][[0, -1]]
  cmin, cmax = np.nonzero(cols)[0][[0, -1]]

  return img[rmin:rmax + 1, cmin:cmax + 1]

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

def centroid(intensity:np.ndarray, x, y, threshold=0/255) -> Tuple[int, int]:
    """
    光强的质心位置

    :param intensity: 强度分布
    :param x: x坐标矩阵
    :param y: y坐标矩阵
    :return center_x, center_y: 光强的质心
    """
    _intensity = intensity.copy()
    _intensity[_intensity < threshold] = 0
    
    total_intensity = np.sum(intensity)
    if total_intensity == 0:
        return 0,0
    
    center_x = np.sum(x * intensity) / total_intensity
    center_y = np.sum(y * intensity) / total_intensity

    return round(center_x), round(center_y)

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

def radius(intensity, x, y, center='centroid', energy=0.99):
    """
    以center为圆心，占总能量百分比为energy的圆的半径

    :param intensity: 强度分布
    :param x: x坐标矩阵
    :param y: y坐标矩阵
    :param center: 圆心，默认'centroid'，可选'peak'，'origin'，坐标，如(0, 0)
    :param energy: 圆内的能量比，默认0.99，取值范围0~1，常用0.5，0.865， 0.99
    :return radius: 圆的半径
    """
    npix = len(x)
    dpix = x[0, 1] - x[0, 0]

    if type(center) is str:
        if center == 'peak':
            x0, y0 = peak_position(intensity, x, y)
        elif center == 'centroid':
            x0, y0 = centroid(intensity, x, y)
        elif center == 'origin':
            x0, y0 = 0, 0
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
            x0, y0 = centroid(intensity, x, y)
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
    center = centroid(img, xv, yv, threshold)
    
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