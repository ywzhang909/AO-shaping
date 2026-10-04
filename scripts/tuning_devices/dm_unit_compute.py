#!/usr/bin/env python3
"""
dmunitcompute.m 的Python实现
用于计算100mm×100mm对应的像素尺寸，并处理stdWavefront目录下的矩阵文件
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from ao_shaping.drivers import MlaRes, ThorlabWFS
from ao_shaping.utils.wavefront.wavefront_calc import (
    ZernikeCentroidCalculator,
    calculate_derotation,
    to_color,
)

# NOTE (R-34, 2026-10-04): `centroid_calculation`,
# `calculate_derotation`, `get_zernike_base_matrixs`, `to_color` and
# `ZernikeCentroidCalculator` used to be duplicated here verbatim. They now
# live only in `ao_shaping.utils.wavefront.wavefront_calc` and are imported
# above. The copy was not merely redundant: its `get_zernike_base_matrixs`
# iterated `Path.glob('*.txt')`, which is lexicographic, so mode 10 sat at
# index 1 -- 65 of the 66 slots were wrong and a pure tilt-x command reported a
# centroid on the array centre instead of a ~35 px shift.
#
# `get_init_V_by_energy` / `get_init_V_by_rms` / `NlightDM` were imported
# but never referenced here; the visualisation loop is WFS-only.


@contextmanager
def visualize_with_pygame(title="Image Visualization"):
    """
    使用pygame实时显示矩阵图像并在图像上显示质心坐标

    参数:
    matrix: 要显示的矩阵
    cx: 质心x坐标
    cy: 质心y坐标
    title: 窗口标题
    """
    # 初始化pygame
    import pygame

    pygame.init()
    calculator = ZernikeCentroidCalculator()
    # 设置窗口大小（根据矩阵大小调整）
    height, width = calculator.shape
    window_width = min(width, 800)  # 最大800像素宽
    window_height = min(height, 600)  # 最大600像素高
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(title)
    scaled_surface = pygame.Surface((width, height))

    # 缩放图像到窗口大小
    scaled_image = pygame.transform.scale(scaled_surface, (window_width, window_height))

    # 字体设置
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    # 创建退出按钮
    button_width = 100
    button_height = 40
    button_x = window_width - button_width - 10
    button_y = 10
    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

    # 主循环
    running = True
    clock = pygame.time.Clock()

    # WFS init
    wfs = ThorlabWFS(MlaRes.Res768)
    wfs.open()

    exposure_time, gain = wfs.optimize_exposure_time_and_gain()
    wfs.exposure_time = exposure_time
    wfs.optimize_pupil()
    wfs.pupil = wfs.optimize_pupil()

    try:
        assert running
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    running = False

        zernike_coef = wfs.get_zernike()
        (dx, dy), img = calculator.get_centroid(zernike_coef)
        cx, cy = calculator.center_coordinate(dx, dy)
        cx, cy = calculate_derotation(cx, cy, np.deg2rad(45))
        # 绘制图像
        scaled_image = pygame.surfarray.make_surface(to_color(img))
        pygame.draw.circle(scaled_image, (255, 0, 0), (int(dx), int(dy)), 10, 15)
        screen.blit(scaled_image, (0, 0))

        # 显示质心坐标
        content = f"center: x={calculator.pix_to_mm(cx):.2f}mm, y={calculator.pix_to_mm(cy):.2f}mm"
        text = font.render(content, True, (255, 255, 255))  # 白色文字
        screen.blit(text, (10, window_height - font.size(content)[1] - 10))

        # 绘制退出按钮
        pygame.draw.rect(screen, (255, 0, 0), button_rect)  # 红色按钮
        button_text = small_font.render("Quit", True, (255, 255, 255))  # 白色文字
        text_rect = button_text.get_rect(center=button_rect.center)
        screen.blit(button_text, text_rect)

        # 更新显示
        pygame.display.flip()
        clock.tick(30)  # 30 FPS
        yield
    except AssertionError as e:
        pass
    finally:
        wfs.close()
        pygame.quit()


if __name__ == "__main__":
    while True:
        visualize_with_pygame()
