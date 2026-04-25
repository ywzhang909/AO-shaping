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