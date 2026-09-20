from enum import Enum


class GxDeviceClassList(Enum):
    UNKNOWN = 0
    USB2 = 1
    GEV = 2
    U3V = 3


class GxAccessStatus(Enum):
    UNKNOWN = 0
    READWRITE = 1
    READONLY = 2
    NOACCESS = 3


class GxAccessMode(Enum):
    READONLY = 2
    CONTROL = 3
    EXCLUSIVE = 4


class GxPixelFormatEntry(Enum):
    UNDEFINED = 0x00000000
    MONO8 = 0x01080001
    MONO8_SIGNED = 0x01080002
    MONO10 = 0x01100003
    MONO12 = 0x01100005
    MONO14 = 0x01100025
    MONO16 = 0x01100007
    BAYER_GR8 = 0x01080008
    BAYER_RG8 = 0x01080009
    BAYER_GB8 = 0x0108000A
    BAYER_BG8 = 0x0108000B
    BAYER_GR10 = 0x0110000C
    BAYER_RG10 = 0x0110000D
    BAYER_GB10 = 0x0110000E
    BAYER_BG10 = 0x0110000F
    BAYER_GR12 = 0x01100010
    BAYER_RG12 = 0x01100011
    BAYER_GB12 = 0x01100012
    BAYER_BG12 = 0x01100013
    BAYER_GR16 = 0x0110002E
    BAYER_RG16 = 0x0110002F
    BAYER_GB16 = 0x01100030
    BAYER_BG16 = 0x01100031
    RGB8_PLANAR = 0x02180021
    RGB10_PLANAR = 0x02300022


class GxFrameStatusList(Enum):
    SUCCESS = 0
    INCOMPLETE = -1


class GxDeviceTemperatureSelectorEntry(Enum):
    SENSOR = 1
    MAINBOARD = 2


class GxPixelSizeEntry(Enum):
    BPP8 = 8
    BPP10 = 10
    BPP12 = 12
    BPP16 = 16
    BPP24 = 24
    BPP30 = 30
    BPP32 = 32
    BPP36 = 36
    BPP48 = 48
    BPP64 = 64


class GxPixelColorFilterEntry(Enum):
    NONE = 0
    BAYER_RG = 1
    BAYER_GB = 2
    BAYER_GR = 3
    BAYER_BG = 4


class GxAcquisitionModeEntry(Enum):
    SINGLE_FRAME = 0
    MULITI_FRAME = 1
    CONTINUOUS = 2


class GxTriggerSourceEntry(Enum):
    SOFTWARE = 0
    LINE0 = 1
    LINE1 = 2
    LINE2 = 3
    LINE3 = 4
    COUNTER2END = 5


class GxTriggerActivationEntry(Enum):
    FALLING_EDGE = 0
    RISING_EDGE = 1


class GxExposureModeEntry(Enum):
    TIMED = 1
    TRIGGER_WIDTH = 2


class GxUserOutputSelectorEntry(Enum):
    OUTPUT0 = 1
    OUTPUT1 = 2
    OUTPUT2 = 4


class GxUserOutputModeEntry(Enum):
    STROBE = 0
    USER_DEFINED = 1


class GxGainSelectorEntry(Enum):
    ALL = 0
    RED = 1
    GREEN = 2
    BLUE = 3


class GxBlackLevelSelectEntry(Enum):
    ALL = 0
    RED = 1
    GREEN = 2
    BLUE = 3


class GxBalanceRatioSelectorEntry(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2


class GxAALightEnvironmentEntry(Enum):
    NATURE_LIGHT = 0
    AC50HZ = 1
    AC60HZ = 2


class GxUserSetEntry(Enum):
    DEFAULT = 0
    USER_SET0 = 1


class GxAWBLampHouseEntry(Enum):
    ADAPTIVE = 0
    D65 = 1
    FLUORESCENCE = 2
    INCANDESCENT = 3
    D75 = 4
    D50 = 5
    U30 = 6


class GxUserDataFieldSelectorEntry(Enum):
    FIELD_0 = 0
    FIELD_1 = 1
    FIELD_2 = 2
    FIELD_3 = 3


class GxTestPatternEntry(Enum):
    OFF = 0
    GRAY_FRAME_RAMP_MOVING = 1
    SLANT_LINE_MOVING = 2
    VERTICAL_LINE_MOVING = 3
    HORIZONTAL_LINE_MOVING = 4
    GREY_VERTICAL_RAMP = 5
    SLANT_LINE = 6


class GxTriggerSelectorEntry(Enum):
    FRAME_START = 1
    FRAME_BURST_START = 2


class GxLineSelectorEntry(Enum):
    LINE0 = 0
    LINE1 = 1
    LINE2 = 2
    LINE3 = 3
    LINE4 = 4
    LINE5 = 5
    LINE6 = 6
    LINE7 = 7
    LINE8 = 8
    LINE9 = 9
    LINE10 = 10
    LINE_STROBE = 11


class GxLineModeEntry(Enum):
    INPUT = 0
    OUTPUT = 1


class GxLineSourceEntry(Enum):
    OFF = 0
    STROBE = 1
    USER_OUTPUT0 = 2
    USER_OUTPUT1 = 3
    USER_OUTPUT2 = 4
    EXPOSURE_ACTIVE = 5
    FRAME_TRIGGER_WAIT = 6
    ACQUISITION_TRIGGER_WAIT = 7
    TIMER1_ACTIVE = 8
    USER_OUTPUT3 = 9
    USER_OUTPUT4 = 10
    USER_OUTPUT5 = 11
    USER_OUTPUT6 = 12


class GxEventSelectorEntry(Enum):
    EXPOSURE_END = 0x0004
    BLOCK_DISCARD = 0x9000
    EVENT_OVERRUN = 0x9001
    FRAME_START_OVER_TRIGGER = 0x9002
    BLOCK_NOT_EMPTY = 0x9003
    INTERNAL_ERROR = 0x9004
    FRAME_BURST_START_OVERT_RIGGER = 0x9005
    FRAME_START_WAIT = 0x9006
    FRAME_BURST_START_WAIT = 0x9007


class GxLutSelectorEntry(Enum):
    LUMINANCE = 0


class GxTransferControlModeEntry(Enum):
    BASIC = 0
    USER_CONTROLED = 1


class GxTransferOperationModeEntry(Enum):
    MULTI_BLOCK = 0


class GxTestPatternGeneratorSelectorEntry(Enum):
    SENSOR = 0
    REGION0 = 1


class GxChunkSelectorEntry(Enum):
    FRAME_ID = 1
    TIME_STAMP = 2
    COUNTER_VALUE = 3


class GxBinningHorizontalModeEntry(Enum):
    SUM = 0
    AVERAGE = 1


class GxBinningVerticalModeEntry(Enum):
    SUM = 0
    AVERAGE = 1


class GxSensorShutterModeEntry(Enum):
    GLOBAL = 0
    ROLLING = 1
    GLOBALRESET = 2


class GxAcquisitionStatusSelectorEntry(Enum):
    ACQUISITION_TRIGGER_WAIT = 0
    FRAME_TRIGGER_WAIT = 1


class GxExposureTimeModeEntry(Enum):
    ULTRASHORT = 0
    STANDARD = 1


class GxGammaModeEntry(Enum):
    SRGB = 0
    USER = 1


class GxLightSourcePresetEntry(Enum):
    OFF = 0
    CUSTOM = 1
    DAYLIGHT_6500K = 2
    DAYLIGHT_5000K = 3
    COOL_WHITE_FLUORESCENCE = 4
    INCA = 5


class GxColorTransformationModeEntry(Enum):
    RGB_TO_RGB = 0
    USER = 1


class GxColorTransformationValueSelectorEntry(Enum):
    GAIN00 = 0
    GAIN01 = 1
    GAIN02 = 2
    GAIN10 = 3
    GAIN11 = 4
    GAIN12 = 5
    GAIN20 = 6
    GAIN21 = 7
    GAIN22 = 8


class GxAutoEntry(Enum):
    OFF = 0
    CONTINUOUS = 1
    ONCE = 2


class GxSwitchEntry(Enum):
    OFF = 0
    ON = 1


class GxRegionSendModeEntry(Enum):
    SINGLE_ROI = 0
    MULTI_ROI = 1


class GxRegionSelectorEntry(Enum):
    REGION0 = 0
    REGION1 = 1
    REGION2 = 2
    REGION3 = 3
    REGION4 = 4
    REGION5 = 5
    REGION6 = 6
    REGION7 = 7


class GxTimerSelectorEntry(Enum):
    TIMER1 = 1


class GxTimerTriggerSourceEntry(Enum):
    EXPOSURE_START = 1
    LINE10 = 10
    STROBE = 16


class GxCounterSelectorEntry(Enum):
    COUNTER1 = 1
    COUNTER2 = 2


class GxCounterEventSourceEntry(Enum):
    FRAME_START = 1
    FRAME_TRIGGER = 2
    ACQUISITION_TRIGGER = 3
    OFF = 4
    SOFTWARE = 5
    LINE0 = 6
    LINE1 = 7
    LINE2 = 8
    LINE3 = 9


class GxCounterResetSourceEntry(Enum):
    OFF = 0
    SOFTWARE = 1
    LINE0 = 2
    LINE1 = 3
    LINE2 = 4
    LINE3 = 5
    COUNTER2END = 6


class GxCounterResetActivationEntry(Enum):
    RISINGEDGE = 1


class GxCounterTriggerSourceEntry(Enum):
    OFF = 0
    SOFTWARE = 1
    LINE0 = 2
    LINE1 = 3
    LINE2 = 4
    LINE3 = 5


class GxTimerTriggerActivationEntry(Enum):
    RISINGEDGE = 1


class GxStopAcquisitionModeEntry(Enum):
    GENERAL = 0
    LIGHT = 1


class GxDSStreamBufferHandlingModeEntry(Enum):
    OLDEST_FIRST = 1
    OLDEST_FIRST_OVERWRITE = 2
    NEWEST_ONLY = 3


class GxResetDeviceModeEntry(Enum):
    RECONNECT = 1
    RESET = 2


class DxBayerConvertType(Enum):
    NEIGHBOUR = 0
    ADAPTIVE = 1
    NEIGHBOUR3 = 2


class DxValidBit(Enum):
    BIT0_7 = 0
    BIT1_8 = 1
    BIT2_9 = 2
    BIT3_10 = 3


class DxImageMirrorMode(Enum):
    HORIZONTAL_MIRROR = 0
    VERTICAL_MIRROR = 1


class DxRGBChannelOrder(Enum):
    ORDER_RGB = 0
    ORDER_BGR = 1
