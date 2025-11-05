import pygame

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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        img_size = img.shape
        canvas = pygame.surfarray.make_surface(img.transpose())
        pygame.draw.circle(canvas, (255, 0, 0), center, r, 3)
        pygame.display.set_caption(info)
        self.window.blit(canvas, (0,0))
        # 绘制电压图
        # 清空之前绘制的条形统计图
        plot_area = pygame.Rect(0, img_size[1], img_size[0], VOLT_HEIGHT)
        self.window.fill(self.background_color, plot_area)
        bar_width = int(img_size[0] / len(volts))
        for i,v in enumerate(volts):
            normed_v = (v - v_min) / (v_max - v_min)
            color = (int(normed_v*255), int((1-normed_v)*255), 0)
            x = int(i * bar_width)
            y = int(img_size[1] + VOLT_HEIGHT)
            height = int(2* v *  VOLT_HEIGHT / (v_max - v_min))
            pygame.draw.line(self.window, color, (x, y), (x, y - height), bar_width)
        
        pygame.event.pump()
        pygame.display.update()

        return True
        
    def close(self) -> None:
        pygame.quit()