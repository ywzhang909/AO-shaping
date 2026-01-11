import pygame
import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.utils import Register

# display settings
VOLT_HEIGHT = 200
LOG_J_HEIGHT = 200
# 定义背景颜色
BACKGROUND_COLOR = (0, 0, 0)
# 定义折线颜色
LINE_COLOR = (0, 255, 0)


class ImageVoltagesDisplay:
    def __init__(self, img_size, volt_height=VOLT_HEIGHT, background_color=BACKGROUND_COLOR):
        self.img_size = img_size
        self.volt_height = volt_height
        self.plot_area = pygame.Rect(0, img_size[1], img_size[0], volt_height)
        self.background_color = background_color
        
    def init_window(self) -> None:
        pygame.init()
        self.window = pygame.display.set_mode((self.img_size[1], self.img_size[0] + self.volt_height*2))

    def render(self, img, volts, v_min, v_max, center, r, info="") -> bool:
        '''
        渲染图像到窗口，同时绘制电压图
        Parameters:
        window (pygame.Surface): 要渲染的窗口
        img (np.ndarray): 要渲染的图像，形状为 (height, width)
        volts (np.ndarray): 电压值数组，用于绘制电压图
        center (tuple): 绘制圆的中心坐标 (x, y)
        r (int): 绘制圆的半径
        info (str, optional): 窗口标题信息，默认值为空字符串
        '''
        if pygame.event.get(pygame.QUIT):
            return False

        img_size = img.shape
        canvas = pygame.surfarray.make_surface(img.transpose())
        pygame.draw.circle(canvas, (255, 0, 0), center, r, 1)
        pygame.display.set_caption(info)
        self.window.blit(canvas, (0,0))
        # 绘制电压图
        # 清空之前绘制的条形统计图
        plot_area = pygame.Rect(0, img_size[1], img_size[0], VOLT_HEIGHT)
        self.window.fill(self.background_color, plot_area)
        bar_width = int(img_size[0] / len(volts))
        for i,v in enumerate(volts):
            normed_v = (v - v_min) / (v_max - v_min)
            color = (int(normed_v*255), int((1-normed_v)*255), 0) if not np.isnan(normed_v) else (0, 0, 0)
            x = int(i * bar_width)
            y = int(img_size[1] + VOLT_HEIGHT)
            height = int(2* v *  VOLT_HEIGHT / (v_max - v_min))
            pygame.draw.line(self.window, color, (x, y), (x, y - height), bar_width)
        
        pygame.event.pump()
        pygame.display.update()

        return True
        
    def close(self) -> None:
        pygame.quit()

plot_funcs = Register()

@plot_funcs.register("voltages")
def plot_voltages(volts, ax:plt.Axes, title="") -> plt.Axes:
    '''
    绘制电压图
    Parameters:
    volts (np.ndarray): 电压值数组，用于绘制电压图
    ax (plt.Axes): 要绘制的Axes对象
    '''
    ax.bar(range(len(volts)), volts)
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Voltage")
    ax.set_axis_off()
    return ax

@plot_funcs.register("img")
def plot_img(img, ax:plt.Axes, title=""):
    '''
    绘制图像
    Parameters:
    img (np.ndarray): 要绘制的图像，形状为 (height, width)
    ax (plt.Axes): 要绘制的Axes对象
    '''
    im = ax.imshow(img, vmin=0, vmax=255, cmap='gray')
    ax.set_title(title)
    ax.set_axis_off()
    return im

@plot_funcs.register("log_j")
def plot_log_j(log_j, ax:plt.Axes, title="") -> plt.Axes:
    '''
    绘制log_j图
    Parameters:
    log_j (np.ndarray): log_j值数组，用于绘制log_j图
    ax (plt.Axes): 要绘制的Axes对象
    '''
    ax.plot(log_j)
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("log_j")
    ax.set_axis_off()
    return ax

@plot_funcs.register("pib_history")
def plot_pib_history(pib_values, ax:plt.Axes, title="PIB History") -> plt.Axes:
    '''
    绘制PIB历史曲线
    Parameters:
    pib_values (list or np.ndarray): PIB值数组，用于绘制历史曲线
    ax (plt.Axes): 要绘制的Axes对象
    title (str): 图标题
    '''
    ax.plot(pib_values)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PIB")
    return ax

@plot_funcs.register("rms_history")
def plot_rms_history(rms_values, ax:plt.Axes, min_epoch=None, min_rms=None, title="RMS History") -> plt.Axes:
    '''
    绘制RMS历史曲线
    Parameters:
    rms_values (list or np.ndarray): RMS值数组，用于绘制历史曲线
    ax (plt.Axes): 要绘制的Axes对象
    min_epoch (int, optional): 最小RMS值对应的epoch
    min_rms (float, optional): 最小RMS值
    title (str): 图标题
    '''
    ax.plot(rms_values)
    if min_epoch is not None and min_rms is not None:
        ax.scatter(min_epoch, min_rms, color='r', marker='*', label='Min RMS')
        ax.text(min_epoch, min_rms, f"{min_rms:.4f}", color='r')
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMS")
    return ax

@plot_funcs.register("wavefront")
def plot_wavefront(wavefront, ax:plt.Axes, title="Wavefront", cmap='gray'):
    '''
    绘制波前图像
    Parameters:
    wavefront (np.ndarray): 波前数据，用于绘制图像
    ax (plt.Axes): 要绘制的Axes对象
    title (str): 图标题
    cmap (str): 颜色映射
    '''
    im = ax.imshow(wavefront, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')
    return im

@plot_funcs.register("voltage_comparison")
def plot_voltage_comparison(init_v, best_v, ax:plt.Axes, title="Voltage Comparison") -> plt.Axes:
    '''
    绘制电压对比柱状图
    Parameters:
    init_v (list or np.ndarray): 初始电压值
    best_v (list or np.ndarray): 最优电压值
    ax (plt.Axes): 要绘制的Axes对象
    title (str): 图标题
    '''
    ax.bar(range(len(init_v)), init_v, color='r', label='Initial', alpha=0.7)
    ax.bar(range(len(best_v)), best_v, color='b', label='Best', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Unit ID")
    ax.set_ylabel("Voltage")
    ax.legend()
    return ax

@plot_funcs.register("voltage_heatmap")
def plot_voltage_heatmap(voltages, ax:plt.Axes, title="Voltage History") -> plt.Axes:
    '''
    绘制电压历史热力图
    Parameters:
    voltages (np.ndarray): 电压历史数据，形状为 (epochs, units)
    ax (plt.Axes): 要绘制的Axes对象
    title (str): 图标题
    '''
    im = ax.imshow(voltages.T, aspect='auto')
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Unit ID")
    plt.colorbar(im, ax=ax)
    return ax