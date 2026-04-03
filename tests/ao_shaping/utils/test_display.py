import pygame
import numpy as np

from ao_shaping.display import AutoDisplay, FrameInfo
from ao_shaping.utils.wavefront_calc import ZernikeCentroidCalculator


def test_autodisplay():
    wavefront = ZernikeCentroidCalculator()
    clock = pygame.time.Clock()
    frames = [
        FrameInfo("fspot", "远场光斑", "Image2DWithBucketFrame"),
        FrameInfo("nspot", "远场光斑", "Image2DFrame"),
        FrameInfo("wf", "波前", "Image2DFrame"),
        FrameInfo("voltage", "波前", "VoltageFrame"),
        FrameInfo("value", "PIB", "LogFrame"),
        FrameInfo("info", "info", "TextFrame"),
    ]
    frames_data = {
        "fspot": {},
        "nspot": {},
        "wf": {},
        "voltage": {},
        "value": {},
        "info": {},
    }
    total_frames = 100_000
    with AutoDisplay(frames) as window:
        for frame_count in range(total_frames):
            frames_data['nspot'] = {'img': np.random.randint(0, 255, (300, 300))}
            coef = np.random.randint(-300, 500, (64,))
            center, wf = wavefront.get_centroid(coef)
            frames_data['fspot'] = {'img': wf, 'center': center, 'r': 10}
            frames_data['wf'] = {'img': wf}
            frames_data['voltage'] = {'volts': coef}
            frames_data['value'] = {'value': np.random.randint(0, 100)}
            frames_data['info'] = {'text': f"Frame {frame_count}/{total_frames}\nPIB: {frames_data['value']['value']}"}
            if not window.render(frame_data=frames_data, info=f"Frame {frame_count}/{total_frames}"):
                break
            clock.tick(60)


def test_display():
    # 初始化 Pygame
    pygame.init()

    IMG_SIZE = (300, 300)
    PLOT_SIZE = (IMG_SIZE[0], 500)
    # 新增折线图区域高度
    LINE_PLOT_HEIGHT = 200
    screen = pygame.display.set_mode(
        (max(IMG_SIZE[0], PLOT_SIZE[0]), (IMG_SIZE[1] + PLOT_SIZE[1] + LINE_PLOT_HEIGHT))
    )
    clock = pygame.time.Clock()  # 用于帧率控制

    # 选择颜色映射
    cmap = pygame.color.THECOLORS
    BAR_WIDTH = PLOT_SIZE[0] // IMG_SIZE[0]
    # 定义背景颜色
    BACKGROUND_COLOR = (0, 0, 0)
    # 定义折线颜色
    LINE_COLOR = (0, 255, 0)
    # 存储历史像素值总和
    historical_pixel_sum = []
    # 最多存储的历史数据点
    MAX_HISTORY = IMG_SIZE[0]

    for _ in range(100):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                print(event.pos)

        img = np.random.randint(0, 255, IMG_SIZE).T
        statistics = np.mean(img, axis=1)
        # 计算当前 img 的像素值总和
        pixel_sum = np.sum(img)
        historical_pixel_sum.append(pixel_sum)
        if len(historical_pixel_sum) > MAX_HISTORY:
            historical_pixel_sum.pop(0)

        img_surface = pygame.surfarray.make_surface(img)
        screen.blit(img_surface, (0, 0))

        # 清空之前绘制的条形统计图
        plot_area = pygame.Rect(0, IMG_SIZE[1], PLOT_SIZE[0], PLOT_SIZE[1])
        screen.fill(BACKGROUND_COLOR, plot_area)

        # 绘制柱状图
        min_value = np.min(statistics)
        max_value = np.max(statistics)
        for i, value in enumerate(statistics):
            # 归一化数值用于颜色映射
            normalized_value = (value - min_value) / (max_value - min_value) if max_value != min_value else 0
            # 选择颜色，这里简单使用灰度
            color = (int(255 * normalized_value), int(255 * normalized_value), int(255 * normalized_value))
            x = i * BAR_WIDTH
            y = IMG_SIZE[1] + PLOT_SIZE[1]
            height = int((value / 255) * PLOT_SIZE[1])
            pygame.draw.line(screen, color, (x, y), (x, y - height), BAR_WIDTH)

        # 清空之前绘制的折线统计图
        line_plot_area = pygame.Rect(0, IMG_SIZE[1] + PLOT_SIZE[1], PLOT_SIZE[0], LINE_PLOT_HEIGHT)
        screen.fill(BACKGROUND_COLOR, line_plot_area)

        # 绘制折线统计图
        if len(historical_pixel_sum) > 1:
            min_sum = min(historical_pixel_sum)
            max_sum = max(historical_pixel_sum)
            points = []
            num_points = len(historical_pixel_sum)
            for i, sum_value in enumerate(historical_pixel_sum):
                # 均匀分布 x 轴坐标
                x = int(i * (PLOT_SIZE[0] / (num_points - 1)))
                y = IMG_SIZE[1] + PLOT_SIZE[1] + LINE_PLOT_HEIGHT - int(
                    (sum_value - min_sum) / (max_sum - min_sum) * LINE_PLOT_HEIGHT
                ) if max_sum != min_sum else IMG_SIZE[1] + PLOT_SIZE[1] + LINE_PLOT_HEIGHT // 2
                points.append((x, y))
            pygame.draw.lines(screen, LINE_COLOR, False, points, 2)

        pygame.display.update()
        # 控制帧率
        clock.tick(30)

def test_swanlab():
    import pandas as pd
    import swanlab
    import json

    df = pd.read_pickle(r"D:\workspace\AO-shaping\data\wf-less\20251112_194659\c8c982a4-d879-4202-b11d-36dda2ed4f41.pkl")

    swanlab.init(
        experiment_name="c8c982a4-d879-4202-b11d-36dda2ed4f41",
        config=json.load(open(r"D:\workspace\AO-shaping\data\wf-less\20251112_194659\c8c982a4-d879-4202-b11d-36dda2ed4f41.json"))
    )
    for i, row in df.iterrows():
        swanlab.log({
            "J": row["J"],
            "PIB": row["value"],
        })