from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import pygame

from ao_shaping.utils.handler import Register

# display settings
VOLT_HEIGHT = 200
LOG_J_HEIGHT = 200
# 定义背景颜色
BACKGROUND_COLOR = (0, 0, 0)
# 定义折线颜色
LINE_COLOR = (0, 255, 0)


class ImageVoltagesDisplay:
    def __init__(self, img_size, volt_height=VOLT_HEIGHT, background_color=BACKGROUND_COLOR):
        import pygame
        self.img_size = img_size
        self.volt_height = volt_height
        self.plot_area = pygame.Rect(0, img_size[1], img_size[0], volt_height)
        self.background_color = background_color

    def init_window(self) -> None:
        import pygame
        pygame.init()
        self.window = pygame.display.set_mode((self.img_size[1], self.img_size[0] + self.volt_height*2))

    def render(self, img, volts, v_min, v_max, center, r, info="") -> bool:
        import pygame
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
        import pygame
        pygame.quit()

plot_funcs = Register()

@plot_funcs.register("voltages")
def plot_voltages(volts, ax, title=""):
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


# Zernike calibration display colors
ZERN_STABLE_COLOR = (0, 255, 0)       # Green: variance < 0.001
ZERN_MODERATE_COLOR = (255, 255, 0)    # Yellow: 0.001 <= variance < 0.01
ZERN_UNSTABLE_COLOR = (255, 0, 0)       # Red: variance >= 0.01
ZERN_BAR_DEFAULT_COLOR = (100, 200, 255)   # Light blue for response bars
ZERN_TEXT_COLOR = (255, 255, 255)     # White text
ZERN_BG_COLOR = (20, 20, 20)         # Dark background
ZERN_PROGRESS_BG = (50, 50, 50)       # Progress bar background
ZERN_PROGRESS_FILL = (0, 200, 255)    # Progress bar fill


class ZernikeCalibrationDisplay:
    '''
    Real-time Zernike response matrix visualization display.

    Displays:
    - Top: Mode info text "Mode X: Zernike(n,m)"
    - Middle-left: Response vector as horizontal bar chart
    - Middle-right: Variance as horizontal bar chart with color coding
    - Bottom: Progress bar showing overall progress
    '''

    def __init__(self, n_wfs_terms: int, n_slm_terms: int, window_size: tuple = (800, 600)) -> None:
        '''
        Initialize the Zernike calibration display.

        Parameters:
        n_wfs_terms: Number of WFS terms (Zernike modes measured by WFS)
        n_slm_terms: Number of SLM terms (Zernike modes applied to SLM)
        window_size: Window dimensions (width, height)
        '''
        self.n_wfs_terms = n_wfs_terms
        self.n_slm_terms = n_slm_terms
        self.window_width = window_size[0]
        self.window_height = window_size[1]

        # Layout constants
        self.header_height = 40
        self.footer_height = 50
        self.chart_area_height = self.window_height - self.header_height - self.footer_height
        self.chart_area_width = self.window_width // 2 - 20

        import pygame
        # Initialize font as None (set in init_window)
        self.font: pygame.font.Font | None = None
        self.title_font: pygame.font.Font | None = None
        self.window: pygame.Surface | None = None
        self.clock = pygame.time.Clock()

    def init_window(self) -> None:
        '''Initialize pygame window for display.'''
        import pygame
        pygame.init()
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Zernike Response Matrix Calibration")

        # Use system fonts - larger for title, regular for info
        self.title_font = pygame.font.SysFont("arial", 28, bold=True)
        self.font = pygame.font.SysFont("consolas", 16)

    def update(
        self,
        mode_index: int,
        mode_name: str,
        response_col: np.ndarray,
        variance_col: np.ndarray,
        current_cycle: int,
        total_cycles: int,
        mean_variance: float
    ) -> bool:
        '''
        Update the display with new measurement data.

        Parameters:
        mode_index: Current mode index (0-based)
        mode_name: Name of the current Zernike mode (e.g., "Zernike(4,2)")
        response_col: Response vector array (n_wfs_terms,) - float64
        variance_col: Variance array (n_wfs_terms,) - float64
        current_cycle: Current calibration cycle (0-based)
        total_cycles: Total number of calibration cycles
        mean_variance: Mean variance across all modes

        Returns:
        True to continue, False if user quit
        '''
        import pygame
        # Handle pygame quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        # Clear window with background color
        self.window.fill(ZERN_BG_COLOR)

        # === TOP: Mode info text ===
        mode_text = self._render_text(
            f"Mode {mode_index + 1}: {mode_name}  |  Cycle {current_cycle + 1}/{total_cycles}  |  Mean Var: {mean_variance:.6f}",
            self.title_font,
            ZERN_TEXT_COLOR
        )
        mode_rect = mode_text.get_rect(center=(self.window_width // 2, self.header_height // 2))
        self.window.blit(mode_text, mode_rect)

        # === MIDDLE-LEFT: Response vector bar chart ===
        self._draw_horizontal_bars(
            response_col,
            (10, self.header_height),
            ZERN_BAR_DEFAULT_COLOR,
            "Response Vector"
        )

        # === MIDDLE-RIGHT: Variance bar chart with color coding ===
        self._draw_variance_bars(
            variance_col,
            (self.window_width // 2 + 10, self.header_height),
            "Variance"
        )

        # === BOTTOM: Progress bar ===
        progress_y = self.window_height - self.footer_height // 2
        total_modes = self.n_slm_terms * total_cycles
        current_progress = mode_index + current_cycle * self.n_slm_terms + 1
        self._draw_progress_bar(
            current_progress,
            total_modes,
            (20, progress_y),
            f"Progress: {current_progress}/{total_modes} modes"
        )

        # Update display
        pygame.display.update()
        self.clock.tick(30)

        return True

    def _render_text(self, text: str, font: pygame.font.Font, color: tuple) -> pygame.Surface:
        '''Render text to a surface.'''
        import pygame
        return font.render(text, True, color)

    def _draw_horizontal_bars(
        self,
        data: np.ndarray,
        origin: tuple,
        bar_color: tuple,
        title: str
    ) -> None:
        '''
        Draw horizontal bar chart.

        Parameters:
        data: Array of values to display (n_wfs_terms,)
        origin: Top-left origin (x, y)
        bar_color: Color for the bars
        title: Chart title
        '''
        x_origin, y_origin = origin

        import pygame

        # Draw title
        title_surf = self._render_text(title, self.font, ZERN_TEXT_COLOR)
        self.window.blit(title_surf, (x_origin, y_origin))

        # Normalize data for display
        data = np.asarray(data, dtype=np.float64)
        max_val = np.max(np.abs(data)) if np.max(np.abs(data)) > 0 else 1.0

        # Chart dimensions
        chart_height = min(self.chart_area_height - 30, 200)
        bar_height = max(3, chart_height // self.n_wfs_terms - 1)
        max_bar_width = self.chart_area_width - 60

        # Draw bars (horizontal, growing from left)
        for i, val in enumerate(data):
            if i >= self.n_wfs_terms:
                break

            # Normalize bar width
            norm_val = abs(val) / max_val
            bar_width = int(norm_val * max_bar_width)
            bar_width = max(1, bar_width)

            y_pos = y_origin + 25 + i * (bar_height + 1)

            # Draw bar
            rect = pygame.Rect(x_origin, y_pos, bar_width, bar_height)
            pygame.draw.rect(self.window, bar_color, rect)

            # Draw value label
            val_text = self._render_text(f"{val:.4f}", self.font, (180, 180, 180))
            self.window.blit(val_text, (x_origin + max_bar_width + 5, y_pos))

    def _draw_variance_bars(
        self,
        data: np.ndarray,
        origin: tuple,
        title: str
    ) -> None:
        '''
        Draw variance bar chart with color coding.

        Color coding:
        - Green (#00FF00): stable (variance < 0.001)
        - Yellow (#FFFF00): moderate (0.001 <= variance < 0.01)
        - Red (#FF0000): unstable (variance >= 0.01)

        Parameters:
        data: Array of variance values (n_wfs_terms,)
        origin: Top-left origin (x, y)
        title: Chart title
        '''
        x_origin, y_origin = origin

        import pygame

        # Draw title
        title_surf = self._render_text(title, self.font, ZERN_TEXT_COLOR)
        self.window.blit(title_surf, (x_origin, y_origin))

        # Convert to numpy array
        data = np.asarray(data, dtype=np.float64)

        # Chart dimensions
        chart_height = min(self.chart_area_height - 30, 200)
        bar_height = max(3, chart_height // self.n_wfs_terms - 1)
        max_bar_width = self.chart_area_width - 60

        # Find max variance for normalization (log scale for better visualization)
        max_var = np.max(data) if np.max(data) > 0 else 1.0
        max_var = max(max_var, 0.01)  # At least show 0.01 scale

        # Draw bars
        for i, var in enumerate(data):
            if i >= self.n_wfs_terms:
                break

            y_pos = y_origin + 25 + i * (bar_height + 1)

            # Determine color based on variance thresholds
            if var < 0.001:
                color = ZERN_STABLE_COLOR
            elif var < 0.01:
                color = ZERN_MODERATE_COLOR
            else:
                color = ZERN_UNSTABLE_COLOR

            # Normalize bar width (log scale)
            norm_log = np.log10(var + 1e-10) / np.log10(max_var + 1e-10)
            bar_width = int(norm_log * max_bar_width)
            bar_width = max(1, bar_width)

            # Draw bar
            rect = pygame.Rect(x_origin, y_pos, bar_width, bar_height)
            pygame.draw.rect(self.window, color, rect)

            # Draw value label
            val_text = self._render_text(f"{var:.6f}", self.font, (200, 200, 200))
            self.window.blit(val_text, (x_origin + max_bar_width + 5, y_pos))

    def _draw_progress_bar(
        self,
        current: int,
        total: int,
        origin: tuple,
        label: str
    ) -> None:
        '''
        Draw progress bar.

        Parameters:
        current: Current progress value
        total: Total value
        origin: Bottom-center origin (x, y)
        label: Label text
        '''
        x_center, y_pos = origin

        import pygame

        # Bar dimensions
        bar_width = self.window_width - 40
        bar_height = 20
        bar_x = 20
        bar_y = y_pos - bar_height // 2

        # Draw background
        bg_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        pygame.draw.rect(self.window, ZERN_PROGRESS_BG, bg_rect)

        # Draw fill
        if total > 0:
            fill_width = int(bar_width * current / total)
            if fill_width > 0:
                fill_rect = pygame.Rect(bar_x, bar_y, fill_width, bar_height)
                pygame.draw.rect(self.window, ZERN_PROGRESS_FILL, fill_rect)

        # Draw border
        pygame.draw.rect(self.window, (100, 100, 100), bg_rect, 2)

        # Draw label
        label_surf = self._render_text(label, self.font, ZERN_TEXT_COLOR)
        label_rect = label_surf.get_rect(center=(self.window_width // 2, bar_y + bar_height + 15))
        self.window.blit(label_surf, label_rect)

    def close(self) -> None:
        '''Close pygame and release resources.'''
        import pygame
        pygame.quit()

    def __enter__(self) -> "ZernikeCalibrationDisplay":
        '''Context manager entry.'''
        self.init_window()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        '''Context manager exit.'''
        self.close()
