from dataclasses import dataclass

@dataclass
class ExposureTime():
    current: int
    _max: int = 100_000
    _min: int = 0
    unit = "ms"
    
    @classmethod
    def build(cls, time_str: str):
        """
        从字符串构建 ExposureTime 对象。

        参数:
        time_str (str): 形如 "50ms" 或 "0.05s" 的字符串，表示曝光时间。

        返回:
        ExposureTime: 构建的 ExposureTime 对象。
        """
        time_str = time_str.strip().lower()
        if time_str.endswith("ms"):
            current = int(float(time_str[:-2]))
        elif time_str.endswith("s"):
            current = int(float(time_str[:-1]) * 1000)
        else:
            raise ValueError("Invalid time string format. Use 'ms' or 's' suffix.")
        return cls(current=current, max=0, min=0,)
    
    def __str__(self) -> str:
        return f"ExposureTime(current={self.current}{self.unit}, max={self.max}{self.unit}, min={self.min}{self.unit})"
    
    @property
    def ms(self):
        return self.current
    
    @ms.setter
    def ms(self, value):
        assert self.min <= value <= self.max, f"Exposure time must be between {self.min}ms and {self.max}ms"
        self.current = int(value)
    
    @property
    def s(self):
        return self.current / 1000

    @s.setter
    def s(self, value):
        self.ms = int(value * 1000)

    @property
    def max(self):
        assert self._max > 0, "Max exposure time must be set"
        return self._max
     
    @max.setter
    def max(self, value):
        assert value > 0, "Max exposure time must be positive"
        self._max = int(value)

    @property
    def min(self):
        assert self._min > 0, "Min exposure time must be set"
        return self._min
     
    @min.setter
    def min(self, value):
        assert value > 0, "Min exposure time must be positive"
        self._min = int(value)
        

@dataclass
class WindowSize():
    width: int
    height: int
    max_width: int
    max_height: int
    
    inc: int