from frames import Image2DFrame, Image2DWithBucketFrame, VoltageFrame
from windows import AutoDisplay, ImageVoltagesDisplay

__FRAME = {
    "Image2D": Image2DFrame,
    "Image2DWithBucket": Image2DWithBucketFrame,
    "Voltage": VoltageFrame,
}

FRAME = __FRAME.keys()

__all__ = [
    "AutoDisplay",
    "ImageVoltagesDisplay",
    "Image2DFrame",
    "Image2DWithBucketFrame",
    "VoltageFrame",
]