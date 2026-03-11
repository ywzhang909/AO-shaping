"""
模拟显示器件
"""
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse

from sim.digitaltwin import utilities as utils


class CCD:
    def __init__(self, npix=512, dpix=2e-3, figsize=(6.5, 6), length_unit='cm', intensity_unit='KW/cm2',
                 bucket_cycle=False, bucket_center=None, bucket_radius=2, cycle_linewidth=2,
                 polar_state=False, n_ellipse=10, ellipse_size=1, polar_linewidth=1,
                 lim=None, cmap='jet', vlim=None):
        """
        CCD绘图

        :param npix: 输出图像的像素数，默认512*512
        :param dpix: 输出图像的像素尺寸，默认2mm
        :param figsize: 画布大小，默认(6.5, 6)
        :param length_unit: 输出图像的坐标单位
        :param intensity_unit: 输出图像的光强单位
        :param bucket_cycle: 是否画桶圆，默认False
        :param bucket_center: 桶圆心，仅bucket_cycle=True启用，默认None
        :param bucket_radius: 桶半径，仅bucket_cycle=True启用，默认None
        :param cycle_linewidth: 桶线宽，仅bucket_cycle=True启用，默认2
        :param polar_state: 是否画偏振态，默认False
        :param n_ellipse: 在整个CCD接收图像上，每行/每列绘制的偏振椭圆的数量，仅polar_state=True启用，默认10
        :param ellipse_size: 绘制椭圆的大小，仅polar_state=True启用，默认1
        :param polar_linewidth: 绘制椭圆的线宽，仅polar_state=True启用，默认1
        :param lim: 截取整个光场的一部分进行显示，默认None表示无截取，(-0.1, 0.1)表示截取10cm*10cm的方形区域显示
        :param cmap: 绘制图形的色图，默认'gray'
        :param vlim: 绘图图形的colobar上下限，默认None自动设置
        """

        self.length_unit = length_unit
        self.intensity_unit = intensity_unit

        self.npix = npix
        self.dpix = dpix
        self.side_length = npix * dpix
        self.x, self.y, self.r, _, _, _ = utils.grid(npix, dpix)
        self.intensity = None

        self.fig, self.ax = None, None
        self.figsize = figsize

        self.bucket_cycle = bucket_cycle
        self.bucket_center = bucket_center
        self.bucket_radius = bucket_radius
        self.cycle_linewidth = cycle_linewidth

        self.polar_state = polar_state
        self.n_ellipse = n_ellipse
        self.ellipse_size = ellipse_size
        self.polar_linewidth = polar_linewidth

        self.lim = lim
        self.cmap = cmap
        self.vlim = vlim

    def get_intensity(self, wave):
        """
        计算CCD接收的光强，并根据CCD的像素和尺寸进行转换

        :return intensity: 光场强度
        """
        if not isinstance(wave, list):
            intensity = wave.intensity
            intensity_trans = utils.matrix_size_trans(intensity, wave.dpix, self.npix, self.dpix)
        else:
            intensity = 0
            for i in range(len(wave)):
                intensity = intensity + wave[i].intensity
            intensity_trans = utils.matrix_size_trans(intensity, wave[0].dpix, self.npix, self.dpix)

        intensity_trans = np.where(intensity_trans >= 0, intensity_trans, 0)
        return intensity_trans

    def bucket_show(self, ax):
        """
        画圆表示桶中功率的桶

        :param ax: 画布的axes
        """

        if self.bucket_center is None or self.bucket_radius is None:
            raise ValueError('The value of buckets center or radius is not set')

        circle = Circle(self.bucket_center, self.bucket_radius, linewidth=self.cycle_linewidth)
        ax.add_patch(circle)

    def polar_show(self, wave, ax):
        """
        在整个CCD面上绘制偏振状态，红色右旋，蓝色左旋，绿色线偏

        :param wave: 波
        :param ax: 画布的axes
        """

        if not isinstance(wave, list):
            if wave.ex is None:
                return None

            wf = utils.matrix_size_trans(wave.wavefront, wave.dpix, self.npix, self.dpix)
            ex = utils.matrix_size_trans(wave.ex, wave.dpix, self.npix, self.dpix)
            ey = utils.matrix_size_trans(wave.ey, wave.dpix, self.npix, self.dpix)

            e = np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(wf) ** 2)
            norm_ex = np.abs(ex) / e.max()
            norm_ey = np.abs(ey) / e.max()
            phase_delay = np.angle(ey) - np.angle(ex)

            d_ellipse = self.side_length / self.n_ellipse
            x = np.linspace(-self.side_length / 2 + d_ellipse / 2,
                            self.side_length / 2 - d_ellipse / 2, self.n_ellipse)
            x = utils.meter2unit(x, self.length_unit)
            y = np.linspace(-self.side_length / 2 + d_ellipse / 2,
                            self.side_length / 2 - d_ellipse / 2, self.n_ellipse)
            y = utils.meter2unit(y, self.length_unit)

            for i in range(self.n_ellipse):
                for j in range(self.n_ellipse):
                    index_row = int(d_ellipse * (i + 0.5) / self.side_length * self.npix)
                    index_column = int(d_ellipse * (j + 0.5) / self.side_length * self.npix)

                    ex_ij = norm_ex[index_row, index_column]
                    ey_ij = norm_ey[index_row, index_column]
                    phase_ij = phase_delay[index_row, index_column]

                    if np.abs(ex_ij * ey_ij * np.sin(phase_ij)) == 0:
                        a = np.sqrt(ex_ij ** 2 + ey_ij ** 2)
                        b = 0
                        if ex_ij == 0:
                            orientation = np.pi / 2
                        else:
                            orientation = np.arctan(ey_ij / ex_ij * np.cos(phase_ij))
                        color = 'green'
                    else:
                        theta = np.arctan(np.tan(phase_ij) * (ex_ij ** 2 - ey_ij ** 2) / (ex_ij ** 2 + ey_ij ** 2)) / 2
                        orientation = np.arctan(
                            ey_ij * np.cos(theta + phase_ij / 2) / ex_ij / np.cos(theta - phase_ij / 2))

                        a = np.sqrt(np.sin(phase_ij) ** 2 / (
                                np.cos(orientation) ** 2 / ex_ij ** 2 + np.sin(orientation) ** 2 / ey_ij ** 2
                                - np.sin(2 * orientation) / ex_ij / ey_ij * np.cos(phase_ij)))
                        b = np.sqrt(np.sin(phase_ij) ** 2 / (
                                np.sin(orientation) ** 2 / ex_ij ** 2 + np.cos(orientation) ** 2 / ey_ij ** 2
                                + np.sin(2 * orientation) / ex_ij / ey_ij * np.cos(phase_ij)))
                        if np.sin(phase_ij) > 0:
                            color = 'red'
                        else:
                            color = 'blue'

                    if a == 0 and b == 0:
                        width, height = 0, 0
                    else:
                        width = self.side_length / self.n_ellipse * self.ellipse_size
                        height = b / a * self.side_length / self.n_ellipse * self.ellipse_size
                    width = utils.meter2unit(width, self.length_unit)
                    height = utils.meter2unit(height, self.length_unit)

                    ellipse = Ellipse((x[j], y[i]), width, height,
                                      angle=orientation / np.pi * 180, edgecolor=color,
                                      linewidth=self.polar_linewidth, fill=False)

                    ax.add_patch(ellipse)
        else:
            raise TypeError('Wave list has not polar state')

    def colorbar(self, image, pad, width, fig, ax):
        """
        设置colorbar

        :param image: 需要设置colorbar的图像
        :param pad: colorbar与ax的间距
        :param width: colorbar的宽度
        :param fig: colorbar的画布
        :param ax: colorbar的axes
        """

        ax_position = ax.get_position()
        cax_position = mpl.transforms.Bbox.from_extents(
            ax_position.x1 + pad, ax_position.y0, ax_position.x1 + pad + width, ax_position.y1)
        cax = ax.figure.add_axes(cax_position)

        cbar = fig.colorbar(image, cax=cax)
        cbar.ax.set_title(self.intensity_unit, fontsize=12, fontfamily='sans-serif', fontstyle='normal')
        cbar.ax.tick_params(labelsize=12)
        cbar.mappable.set_clim(0)

    def out(self, wave, save=False, save_dir=None):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        plt.subplots_adjust(left=0.05)

        self.intensity = self.get_intensity(wave)

        x = utils.meter2unit(self.x, self.length_unit)
        y = utils.meter2unit(self.y, self.length_unit)
        intensity = utils.w_sqr_m2unit(self.intensity, self.intensity_unit)

        if self.vlim is None:
            im = self.ax.pcolor(x, y, intensity, cmap=self.cmap)
        else:
            im = self.ax.pcolor(x, y, intensity, cmap=self.cmap, vmin=self.vlim[0], vmax=self.vlim[1])
        self.ax.set_xlabel(self.length_unit, fontsize=12, fontfamily='sans-serif', fontstyle='normal')
        self.ax.set_ylabel(self.length_unit, fontsize=12, fontfamily='sans-serif', fontstyle='normal', labelpad=-5)
        self.ax.set_aspect('equal')
        self.ax.xaxis.set_tick_params(labelsize=12)
        self.ax.yaxis.set_tick_params(labelsize=12)

        if self.lim is not None:
            try:
                if self.lim[1] - self.lim[0] > self.side_length:
                    raise Warning('Observation area is larger than CCD side length')
            except Warning:
                print('Observation area is larger than CCD side length')

            bottom = utils.meter2unit(self.lim[0], self.length_unit)
            top = utils.meter2unit(self.lim[1], self.length_unit)
            self.ax.set_xlim(bottom, top)
            self.ax.set_ylim(bottom, top)

        self.colorbar(im, 0.02, 0.04, self.fig, self.ax)

        if self.bucket_cycle:
            self.bucket_show(self.ax)
        if self.polar_state:
            self.polar_show(wave, self.ax)

        if save:
            self.fig.savefig(save_dir, dpi=600, facecolor='white')
            plt.close(self.fig)


class TimeCCD(CCD):
    def __init__(self, display_time, delta_time, display_screen, npix=512, dpix=2e-3, figsize=(6.5, 6),
                 length_unit='cm', intensity_unit='KW/cm2',
                 bucket_cycle=False, bucket_center=None, bucket_radius=2, cycle_linewidth=2,
                 polar_state=False, n_ellipse=10, ellipse_size=1, polar_linewidth=1,
                 lim=None, cmap='jet'):
        """
        CCD绘图

        :param display_time: 需要展示的时间，例如np.linspace(0, 1, 11)表示0s，0.1s...1.0s
        :param delta_time: 大气传输计算的时间间隔
        :param npix: 输出图像的像素数，默认512*512
        :param dpix: 输出图像的像素尺寸，默认2mm
        :param figsize: 画布大小，默认(6.5, 6)
        :param length_unit: 输出图像的坐标单位
        :param intensity_unit: 输出图像的光强单位
        :param bucket_cycle: 是否画桶圆，默认False
        :param bucket_center: 桶圆心，仅bucket_cycle=True启用，默认None
        :param bucket_radius: 桶半径，仅bucket_cycle=True启用，默认None
        :param cycle_linewidth: 桶线宽，仅bucket_cycle=True启用，默认2
        :param polar_state: 是否画偏振态，默认False
        :param n_ellipse: 在整个CCD接收图像上，每行/每列绘制的偏振椭圆的数量，仅polar_state=True启用，默认10
        :param ellipse_size: 绘制椭圆的大小，仅polar_state=True启用，默认1
        :param polar_linewidth: 绘制椭圆的线宽，仅polar_state=True启用，默认1
        :param lim: 截取整个光场的一部分进行显示，默认None表示无截取，(-0.1, 0.1)表示截取10cm*10cm的方形区域显示
        :param cmap: 绘制图形的色图，默认'gray'
        """

        super(TimeCCD, self).__init__(npix, dpix, figsize, length_unit, intensity_unit,
                 bucket_cycle, bucket_center, bucket_radius, cycle_linewidth,
                 polar_state, n_ellipse, ellipse_size, polar_linewidth,
                 lim, cmap)

        self.delta_time = delta_time
        self.display_time = display_time
        self.display_screen = display_screen

        self.fig, self.ax = [], []

    def bucket_show(self, ax):
        """
        画圆表示桶中功率的桶

        :param ax: 画布的axes
        """

        if self.bucket_center is None or self.bucket_radius is None:
            raise ValueError('The value of buckets center or radius is not set')

        circle = Circle(self.bucket_center, self.bucket_radius, linewidth=self.cycle_linewidth)
        ax.add_patch(circle)

    def polar_show(self, wave, ax):
        """
        在整个CCD面上绘制偏振状态，红色右旋，蓝色左旋，绿色线偏

        :param wave: 波
        :param ax: 画布的axes
        """

        if not isinstance(wave, list):
            e = np.sqrt(np.abs(wave.ex) ** 2 + np.abs(wave.ey) ** 2 + np.abs(wave.wavefront) ** 2)
            norm_ex = np.abs(wave.ex) / e.max()
            norm_ey = np.abs(wave.ey) / e.max()
            phase_delay = np.angle(wave.ey) - np.angle(wave.ex)

            d_ellipse = self.side_length / self.n_ellipse
            x = np.linspace(-self.side_length / 2 + d_ellipse / 2, self.side_length / 2 - d_ellipse / 2, self.n_ellipse)
            x = utils.meter2unit(x, self.length_unit)
            y = np.linspace(-self.side_length / 2 + d_ellipse / 2, self.side_length / 2 - d_ellipse / 2, self.n_ellipse)
            y = utils.meter2unit(y, self.length_unit)

            for i in range(self.n_ellipse):
                for j in range(self.n_ellipse):
                    index_row = int(d_ellipse / self.side_length * self.npix * (i + 0.5))
                    index_column = int(d_ellipse / self.side_length * self.npix * (j + 0.5))

                    ex_ij = norm_ex[index_row, index_column]
                    ey_ij = norm_ey[index_row, index_column]
                    phase_ij = phase_delay[index_row, index_column]

                    if np.abs(ex_ij * ey_ij * np.sin(phase_ij)) < 1e-10:
                        a = np.sqrt(ex_ij ** 2 + ey_ij ** 2)
                        b = 0
                        if ex_ij == 0:
                            orientation = np.pi / 2
                        else:
                            orientation = np.arctan(ey_ij / ex_ij * np.cos(phase_ij))
                        color = 'green'
                    else:
                        theta = np.arctan(np.tan(phase_ij) * (ex_ij ** 2 - ey_ij ** 2) / (ex_ij ** 2 + ey_ij ** 2)) / 2
                        orientation = np.arctan(
                            ey_ij * np.cos(theta + phase_ij / 2) / ex_ij / np.cos(theta - phase_ij / 2))

                        a = np.sqrt(np.sin(phase_ij) ** 2 / (
                                np.cos(orientation) ** 2 / ex_ij ** 2 + np.sin(orientation) ** 2 / ey_ij ** 2
                                - np.sin(2 * orientation) / ex_ij / ey_ij * np.cos(phase_ij)))
                        b = np.sqrt(np.sin(phase_ij) ** 2 / (
                                np.sin(orientation) ** 2 / ex_ij ** 2 + np.cos(orientation) ** 2 / ey_ij ** 2
                                + np.sin(2 * orientation) / ex_ij / ey_ij * np.cos(phase_ij)))
                        if np.sin(phase_ij) > 0:
                            color = 'red'
                        else:
                            color = 'blue'

                    ellipse = Ellipse((x[j], y[i]), a * 5 * self.ellipse_size, b * 5 * self.ellipse_size,
                                      angle=orientation / np.pi * 180, edgecolor=color,
                                      linewidth=self.polar_linewidth, fill=False)

                    ax.add_patch(ellipse)
        else:
            raise TypeError('Wave list has not polar state')

    def colorbar(self, image, pad, width, fig, ax):
        """
        设置colorbar

        :param image: 需要设置colorbar的图像
        :param pad: colorbar与ax的间距
        :param width: colorbar的宽度
        :param fig: colorbar的画布
        :param ax: colorbar的axes
        """

        ax_position = ax.get_position()
        cax_position = mpl.transforms.Bbox.from_extents(
            ax_position.x1 + pad, ax_position.y0, ax_position.x1 + pad + width, ax_position.y1)
        cax = ax.figure.add_axes(cax_position)

        cbar = fig.colorbar(image, cax=cax)
        cbar.ax.set_title(self.intensity_unit, fontsize=12, fontfamily='sans-serif', fontstyle='normal')
        cbar.ax.tick_params(labelsize=12)

    def out(self, wave):
        for i in range(len(self.display_time)):
            fig, ax = plt.subplots(figsize=self.figsize)
            plt.subplots_adjust(left=0.05)
            self.fig.append(fig)
            self.ax.append(ax)

        x = utils.meter2unit(self.x, self.length_unit)
        y = utils.meter2unit(self.y, self.length_unit)

        for i in range(len(self.display_time)):
            time = round(self.display_time[i] / self.delta_time) * self.delta_time
            with open('temp/wave l{0:d} t{1:.6f}.pkl'.format(self.display_screen, time), 'rb') as file:
                wave = pickle.load(file)

            intensity = self.get_intensity(wave)
            intensity = utils.w_sqr_m2unit(intensity, self.intensity_unit)
            im = self.ax[i].pcolor(x, y, intensity, cmap=self.cmap)
            self.ax[i].set_xlabel(self.length_unit, fontsize=12, fontfamily='sans-serif', fontstyle='normal')
            self.ax[i].set_ylabel(
                self.length_unit, fontsize=12, fontfamily='sans-serif', fontstyle='normal', labelpad=-5)
            self.ax[i].set_aspect('equal')
            self.ax[i].xaxis.set_tick_params(labelsize=12)
            self.ax[i].yaxis.set_tick_params(labelsize=12)
            self.ax[i].set_title('time: {:.3f} s'.format(self.display_time[i]), fontsize=16)

            if self.lim is not None:
                try:
                    if self.lim[1] - self.lim[0] > self.side_length:
                        raise Warning('Observation area is larger than CCD side length')
                except Warning:
                    print('Observation area is larger than CCD side length')

                bottom = utils.meter2unit(self.lim[0], self.length_unit)
                top = utils.meter2unit(self.lim[1], self.length_unit)
                self.ax[i].set_xlim(bottom, top)
                self.ax[i].set_ylim(bottom, top)

            self.colorbar(im, 0.02, 0.04, self.fig[i], self.ax[i])

            if self.bucket_cycle:
                self.bucket_show(self.ax[i])
            if self.polar_state:
                self.polar_show(wave, self.ax[i])

            if not os.path.isdir('Figure'):
                os.mkdir('Figure')
            self.fig[i].savefig('Figure/Time {0:.3f} s.png'.format(self.display_time[i]))


class PhaseDetector:
    def __init__(self, npix=512, dpix=2e-3, figsize=(6.5, 6), length_unit='cm',
                 bucket_cycle=False, bucket_center=None, bucket_radius=2, cycle_linewidth=2,
                 lim=None, cmap='jet'):
        """
        相位绘图

        :param npix: 输出图像的像素数，默认512*512
        :param dpix: 输出图像的像素尺寸，默认2mm
        :param figsize: 画布大小，默认(6.5, 6)
        :param length_unit: 输出图像的坐标单位
        :param bucket_cycle: 是否画桶圆，默认False
        :param bucket_center: 桶圆心，仅bucket_cycle=True启用，默认None
        :param bucket_radius: 桶半径，仅bucket_cycle=True启用，默认None
        :param cycle_linewidth: 桶线宽，仅bucket_cycle=True启用，默认2
        :param lim: 截取整个光场的一部分进行显示，默认None表示无截取，(-0.1, 0.1)表示截取10cm*10cm的方形区域显示
        :param cmap: 绘制图形的色图，默认'gray'
        """

        self.length_unit = length_unit

        self.npix = npix
        self.dpix = dpix
        self.side_length = npix * dpix
        self.x, self.y, self.r, _, _, _ = utils.grid(npix, dpix)
        self.phase = None

        self.fig, self.ax = None, None
        self.figsize = figsize

        self.bucket_cycle = bucket_cycle
        self.bucket_center = bucket_center
        self.bucket_radius = bucket_radius
        self.cycle_linewidth = cycle_linewidth

        self.lim = lim
        self.cmap = cmap

    def get_phase(self, wave):
        """
        从wave获取相位，调整大小

        :param wave: 波
        :return phase_trans: 转换后相位
        """

        if not isinstance(wave, list):
            phase = np.angle(wave.wavefront)
            if wave.dpix != self.dpix or len(phase) != self.npix:
                phase = utils.matrix_size_trans(phase, wave.dpix, self.npix, self.dpix)
        else:
            raise TypeError('wave list has no phase')

        return phase

    def bucket_show(self, ax):
        """
        画圆表示桶中功率的桶

        :param ax: 画布的axes
        """

        if self.bucket_center is None or self.bucket_radius is None:
            raise ValueError('The value of buckets center or radius is not set')

        circle = Circle(self.bucket_center, self.bucket_radius, linewidth=self.cycle_linewidth)
        ax.add_patch(circle)

    @staticmethod
    def colorbar(image, pad, width, fig, ax):
        """
        设置colorbar

        :param image: 需要设置colorbar的图像
        :param pad: colorbar与ax的间距
        :param width: colorbar的宽度
        :param fig: colorbar的画布
        :param ax: colorbar的axes
        """

        ax_position = ax.get_position()
        cax_position = mpl.transforms.Bbox.from_extents(
            ax_position.x1 + pad, ax_position.y0, ax_position.x1 + pad + width, ax_position.y1)
        cax = ax.figure.add_axes(cax_position)

        cbar = fig.colorbar(image, cax=cax)
        cbar.ax.set_title('rad', fontsize=12, fontfamily='sans-serif', fontstyle='normal')
        cbar.ax.tick_params(labelsize=12)

    def out(self, phase):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        plt.subplots_adjust(left=0.05)

        self.phase = phase

        x = utils.meter2unit(self.x, self.length_unit)
        y = utils.meter2unit(self.y, self.length_unit)

        im = self.ax.pcolor(x, y, self.phase, cmap=self.cmap)
        self.ax.set_xlabel(self.length_unit, fontsize=12, fontfamily='sans-serif', fontstyle='normal')
        self.ax.set_ylabel(self.length_unit, fontsize=12, fontfamily='sans-serif', fontstyle='normal', labelpad=-5)
        self.ax.set_aspect('equal')
        self.ax.xaxis.set_tick_params(labelsize=12)
        self.ax.yaxis.set_tick_params(labelsize=12)

        if self.lim is not None:
            try:
                if self.lim[1] - self.lim[0] > self.side_length:
                    raise Warning('Observation area is larger than CCD side length')
            except Warning:
                print('Observation area is larger than CCD side length')

            bottom = utils.meter2unit(self.lim[0], self.length_unit)
            top = utils.meter2unit(self.lim[1], self.length_unit)
            self.ax.set_xlim(bottom, top)
            self.ax.set_ylim(bottom, top)

        self.colorbar(im, 0.02, 0.04, self.fig, self.ax)

        if self.bucket_cycle:
            self.bucket_show(self.ax)


class LinePlot:
    def __init__(self, *args, x_label, y_label, figsize=(6, 6), log_axis=(False, False)):
        self.xy = list(args)
        self.x_label, self.y_label = x_label, y_label
        self.figsize = figsize
        self.fig, self.ax = None, None
        self.log_axis = log_axis

    def out(self):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        plt.subplots_adjust(left=0.15, right=0.90, bottom=0.12, top=0.90)

        for i in range(len(self.xy) // 2):
            self.ax.plot(self.xy[i*2], self.xy[i*2+1])
        self.ax.set_xlabel(self.x_label, fontsize=12, fontfamily='sans-serif', fontstyle='normal')
        self.ax.set_ylabel(self.y_label, fontsize=12, fontfamily='sans-serif', fontstyle='normal', labelpad=0)
        self.ax.xaxis.set_tick_params(labelsize=12)
        self.ax.yaxis.set_tick_params(labelsize=12)

        if self.log_axis[0]:
            self.ax.set_xscale('log')
        if self.log_axis[1]:
            self.ax.set_yscale('log')

        self.ax.set_xlim(min(self.xy[0]), max(self.xy[0]))
