import ctypes
from ctypes import (
    POINTER,
    byref,
    c_bool,
    c_char,
    c_char_p,
    c_double,
    c_float,
    c_int16,
    c_int32,
    c_uint8,
    c_ulong,
    cdll,
    create_string_buffer,
)

import numpy as np

# WFS status bits
WFS_STATUS = {
    "CON": 0x00000001,  # USB connection lost, set by driver
    "PTH": 0x00000002,  # Power too high (cam saturated)
    "PTL": 0x00000004,  # Power too low (low cam digits)
    "HAL": 0x00000008,  # High ambient light
    "SCL": 0x00000010,  # Spot contrast too low
    "ZFL": 0x00000020,  # Zernike fit failed because of not enough detected spots
    "ZFH": 0x00000040,  # Zernike fit failed because of too much detected spots
    "ATR": 0x00000080,  # Camera is still awaiting a trigger
    "CFG": 0x00000100,  # Camera is configured, ready to use
    "PUD": 0x00000200,  # Pupil is defined
    "SPC": 0x00000400,  # No. of spots or pupil or aoi has been changed
    "RDA": 0x00000800,  # Reconstructed spot deviations available
    "URF": 0x00001000,  # User reference data available
    "HSP": 0x00002000,  # Camera is in Highspeed Mode
    "MIS": 0x00004000,  # Mismatched centroids in Highspeed Mode
    "LOS": 0x00008000,  # low number of detected spots, warning: reduced Zernike accuracy
    "FIL": 0x00010000,  # pupil is badly filled with spots, warning: reduced Zernike accuracy
}


# MAX_SPOTS is actually a constrained by the library version
# see WFS.h for the actual value
MAX_SPOTS = [80, 80]

# defining names according to the manual
ViStatus = c_int32
ViBoolean = c_bool
ViSession = c_ulong
ViUInt8 = c_uint8
ViInt16 = c_int16
ViInt32 = c_int32
ViReal32 = c_float
ViReal64 = c_double
ViChar256 = c_char * 256
ViChar512 = c_char * 512
ViRsrc = ViChar256
ArrFloat = np.ctypeslib.ndpointer(shape=MAX_SPOTS[::-1])  # note the Y, X order
ArrImg = np.ctypeslib.ndpointer(dtype=np.uint8, shape=(512, 512))


# VI_NULL = lambda: None
def VI_NULL():
    return c_ulong()


def to_int(hexvar):
    """
    This function returns hex thing like  b"0x00000400" to integers
    """
    return int(hexvar, 0)


class WfsError(Exception):
    pass

def load_dll():
    dll = cdll.LoadLibrary(r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\WFS_64.dll")
    dll.WFS_init.restype = ViStatus
    dll.WFS_init.argtypes = [ViRsrc, ViBoolean, ViBoolean, POINTER(ViSession)]

    dll.WFS_close.restype = ViStatus
    dll.WFS_close.argtypes = [ViSession]

    # configuration functions
    dll.WFS_GetInstrumentInfo.restype = ViStatus
    dll.WFS_GetInstrumentInfo.argtypes = [
        ViSession,
        ViChar256,
        ViChar256,
        ViChar256,
        ViChar256,
    ]
    # ViStatus __fastcall WFS_GetInstrumentListInfo(ViSession instrumentHandle, ViInt32 instrumentListIndex, ViInt32 *deviceID, ViInt32 *inUse, ViChar *instrumentName, ViChar *instrumentSN, ViChar *resourceName)
    dll.WFS_GetInstrumentListInfo.restype = ViStatus
    dll.WFS_GetInstrumentListInfo.argtypes = [
        ViSession,
        ViInt32,
        POINTER(ViInt32),
        POINTER(ViInt32),
        ViRsrc,
        ViRsrc,
        ViRsrc,
    ]

    dll.WFS_ConfigureCam.restype = ViStatus
    dll.WFS_ConfigureCam.argtypes = [
        ViSession,
        ViInt32,
        ViInt32,
        POINTER(ViInt32),
        POINTER(ViInt32),
    ]

    dll.WFS_SetHighspeedMode.restype = ViStatus
    dll.WFS_SetHighspeedMode.argtypes = [ViSession, ViInt32, ViInt32, ViInt32, ViInt32]

    # dll.WFS_GetHighspeedWindows.restype = ViStatus
    # dll.WFS_GetHighspeedWindows.argtypes = [ViSession, POINTER(ViInt32), POINTER(ViInt32), POINTER(ViInt32), POINTER(ViInt32), ViInt32, ViInt32]

    dll.WFS_CheckHighspeedCentroids.restype = ViStatus
    dll.WFS_CheckHighspeedCentroids.argtypes = [ViSession]

    dll.WFS_GetExposureTimeRange.restype = ViStatus
    dll.WFS_GetExposureTimeRange.argtypes = [
        ViSession,
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
    ]

    dll.WFS_SetExposureTime.restype = ViStatus
    dll.WFS_SetExposureTime.argtypes = [ViSession, ViReal64, POINTER(ViReal64)]

    dll.WFS_GetExposureTime.restype = ViStatus
    dll.WFS_GetExposureTime.argtypes = [ViSession, POINTER(ViReal64)]

    dll.WFS_GetMasterGainRange.restype = ViStatus
    dll.WFS_GetMasterGainRange.argtypes = [
        ViSession,
        POINTER(ViReal64),
        POINTER(ViReal64),
    ]

    dll.WFS_SetMasterGain.restype = ViStatus
    dll.WFS_SetMasterGain.argtypes = [ViSession, ViReal64, POINTER(ViReal64)]

    dll.WFS_GetMasterGain.restype = ViStatus
    dll.WFS_GetMasterGain.argtypes = [ViSession, POINTER(ViReal64)]

    dll.WFS_SetBlackLevelOffset.restype = ViStatus
    dll.WFS_SetBlackLevelOffset.argtypes = [ViSession, ViInt32]

    dll.WFS_GetBlackLevelOffset.restype = ViStatus
    dll.WFS_GetBlackLevelOffset.argtypes = [ViSession, POINTER(ViInt32)]

    dll.WFS_SetTriggerMode.restype = ViStatus
    dll.WFS_SetTriggerMode.argtypes = [ViSession, ViInt32]

    dll.WFS_GetTriggerMode.restype = ViStatus
    dll.WFS_GetTriggerMode.argtypes = [ViSession, POINTER(ViInt32)]

    # dll.WFS_SetTriggerDelayRange.restype = ViStatus
    # dll.WFS_SetTriggerDelayRange.argtypes = [ViSession, POINTER(ViInt32), POINTER(ViInt32), POINTER(ViInt32)]

    dll.WFS_GetMlaCount.restype = ViStatus
    dll.WFS_GetMlaCount.argtypes = [ViSession, POINTER(ViInt32)]

    dll.WFS_GetMlaData.restype = ViStatus
    dll.WFS_GetMlaData.argtypes = [
        ViSession,
        ViInt32,
        ViChar256,
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
    ]

    dll.WFS_GetMlaData2.restype = ViStatus
    dll.WFS_GetMlaData2.argtypes = [
        ViSession,
        ViInt32,
        ViChar256,
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
    ]

    dll.WFS_SelectMla.restype = ViStatus
    dll.WFS_SelectMla.argtypes = [ViSession, ViInt32]

    # WFS_SetAoi and WFS_SetAoi are undocumented and thus left out

    dll.WFS_SetPupil.restype = ViStatus
    dll.WFS_SetPupil.argtypes = [ViSession, ViReal64, ViReal64, ViReal64, ViReal64]

    dll.WFS_GetPupil.restype = ViStatus
    dll.WFS_GetPupil.argtypes = [
        ViSession,
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
    ]

    dll.WFS_SetReferencePlane.restype = ViStatus
    dll.WFS_SetReferencePlane.argtypes = [ViSession, ViInt32]

    dll.WFS_GetReferencePlane.restype = ViStatus
    dll.WFS_GetReferencePlane.argtypes = [ViSession, POINTER(ViInt32)]

    # Action/Status Functions
    dll.WFS_GetStatus.restype = ViStatus
    dll.WFS_GetStatus.argtypes = [ViSession, POINTER(ViInt32)]

    # Data Functions
    dll.WFS_TakeSpotfieldImage.restype = ViStatus
    dll.WFS_TakeSpotfieldImage.argtypes = [ViSession]

    dll.WFS_TakeSpotfieldImageAutoExpos.restype = ViStatus
    dll.WFS_TakeSpotfieldImageAutoExpos.argtypes = [
        ViSession,
        POINTER(ViReal64),
        POINTER(ViReal64),
    ]

    # WFS_GetSpotfieldImage left out

    dll.WFS_GetSpotfieldImageCopy.restype = ViStatus
    dll.WFS_GetSpotfieldImageCopy.argtypes = [
        ViSession,
        POINTER(c_uint8),
        POINTER(ViInt32),
        POINTER(ViInt32),
    ]  # ViUInt8[]

    dll.WFS_AverageImage.restype = ViStatus
    dll.WFS_AverageImage.argtypes = [ViSession, ViInt32, POINTER(ViInt32)]

    dll.WFS_AverageImageRolling.restype = ViStatus
    dll.WFS_AverageImageRolling.argtypes = [ViSession, ViInt32, ViInt32]

    dll.WFS_CutImageNoiseFloor.restype = ViStatus
    dll.WFS_CutImageNoiseFloor.argtypes = [ViSession, ViInt32]

    dll.WFS_CalcImageMinMax.restype = ViStatus
    dll.WFS_CalcImageMinMax.argtypes = [
        ViSession,
        POINTER(ViInt32),
        POINTER(ViInt32),
        POINTER(ViReal64),
    ]

    dll.WFS_CalcMeanRmsNoise.restype = ViStatus
    dll.WFS_CalcMeanRmsNoise.argtypes = [
        ViSession,
        POINTER(ViReal64),
        POINTER(ViReal64),
    ]

    dll.WFS_GetLine.restype = ViStatus
    dll.WFS_GetLine.argtypes = [ViSession, ViInt32, c_float]  # float[]

    dll.WFS_GetLineView.restype = ViStatus
    dll.WFS_GetLineView.argtypes = [ViSession, c_float, c_float]  # float[]

    dll.WFS_CalcBeamCentroidDia.restype = ViStatus
    dll.WFS_CalcBeamCentroidDia.argtypes = [
        ViSession,
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
    ]

    dll.WFS_CalcSpotsCentrDiaIntens.restype = ViStatus
    dll.WFS_CalcSpotsCentrDiaIntens.argtypes = [ViSession, ViInt32, ViInt32]

    dll.WFS_GetSpotCentroids.restype = ViStatus
    dll.WFS_GetSpotCentroids.argtypes = [ViSession, ArrFloat, ArrFloat]  # float[]

    dll.WFS_GetSpotDiameters.restype = ViStatus
    dll.WFS_GetSpotDiameters.argtypes = [ViSession, c_float, c_float]  # float[]

    dll.WFS_GetSpotDiaStatistics.restype = ViStatus
    dll.WFS_GetSpotDiaStatistics.argtypes = [
        ViSession,
        POINTER(ViInt32),
        POINTER(ViInt32),
        POINTER(ViInt32),
    ]

    dll.WFS_GetSpotIntensities.restype = ViStatus
    dll.WFS_GetSpotIntensities.argtypes = [ViSession, ArrFloat]  # float[]

    dll.WFS_CalcSpotToReferenceDeviations.restype = ViStatus
    dll.WFS_CalcSpotToReferenceDeviations.argtypes = [ViSession, ViInt32]

    dll.WFS_GetSpotReferencePositions.restype = ViStatus
    dll.WFS_GetSpotReferencePositions.argtypes = [
        ViSession,
        c_float,
        c_float,
    ]  # float[]

    dll.WFS_GetSpotDeviations.restype = ViStatus
    dll.WFS_GetSpotDeviations.argtypes = [ViSession, ArrFloat, ArrFloat]  # float[]

    # dll.WFS_ZernikeLsf.restype = ViStatus
    # dll.WFS_ZernikeLsf.argtypes = [ViSession, POINTER(ViInt32), c_float, c_float, POINTER(ViReal64)] # float[]

    dll.WFS_CalcFourierOptometric.restype = ViStatus
    dll.WFS_CalcFourierOptometric.argtypes = [
        ViSession,
        ViInt32,
        ViInt32,
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
    ]

    dll.WFS_CalcReconstrDeviations.restype = ViStatus
    dll.WFS_CalcReconstrDeviations.argtypes = [
        ViSession,
        ViInt32,
        ViInt32,
        ViInt32,
        POINTER(ViReal64),
        POINTER(ViReal64),
    ]  # ViInt32[]

    dll.WFS_CalcWavefront.restype = ViStatus
    dll.WFS_CalcWavefront.argtypes = [ViSession, ViInt32, ViInt32, ArrFloat]  # float[]

    dll.WFS_CalcWavefrontStatistics.restype = ViStatus
    dll.WFS_CalcWavefrontStatistics.argtypes = [
        ViSession,
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
        POINTER(ViReal64),
    ]

    # Utility Functions
    dll.WFS_self_test.restype = ViStatus
    dll.WFS_self_test.argtypes = [ViSession, ViInt16, c_char_p]  # ViChar[]

    dll.WFS_reset.restype = ViStatus
    dll.WFS_reset.argtypes = [ViSession]

    dll.WFS_revision_query.restype = ViStatus
    dll.WFS_revision_query.argtypes = [ViSession, c_char_p, c_char_p]  # ViChar[]

    dll.WFS_error_query.restype = ViStatus
    dll.WFS_error_query.argtypes = [ViSession, POINTER(ViInt32), c_char_p]  # ViChar[]

    dll.WFS_error_message.restype = ViStatus
    dll.WFS_error_message.argtypes = [ViSession, ViStatus, POINTER(ViChar256)]

    dll.WFS_GetInstrumentListLen.restype = ViStatus
    dll.WFS_GetInstrumentListLen.argtypes = [ViSession, POINTER(ViInt32)]

    dll.WFS_GetInstrumentListInfo.restype = ViStatus
    dll.WFS_GetInstrumentListInfo.argtypes = [
        ViSession,
        ViInt32,
        POINTER(ViInt32),
        POINTER(ViInt32),
        ViChar256,
        ViChar256,
        ViRsrc,
    ]  # ViChar[]

    dll.WFS_GetXYScale.restype = ViStatus
    dll.WFS_GetXYScale.argtypes = [ViSession, c_float, c_float]  # float[]

    dll.WFS_ConvertWavefrontWaves.restype = ViStatus
    dll.WFS_ConvertWavefrontWaves.argtypes = [
        ViSession,
        ViReal64,
        ViReal32,
        ViReal32,
    ]  # ViReal[]

    dll.WFS_Flip2DArray.restype = ViStatus
    dll.WFS_Flip2DArray.argtypes = [ViSession, ViReal32, ViReal32]  # ViReal32

    # Calibration Functions
    dll.WFS_SetSpotsToUserReference.restype = ViStatus
    dll.WFS_SetSpotsToUserReference.argtypes = [ViSession]

    dll.WFS_SetCalcSpotsToUserReference.restype = ViStatus
    dll.WFS_SetCalcSpotsToUserReference.argtypes = [
        ViSession,
        ViInt32,
        c_float,
        c_float,
    ]  # float[]

    dll.WFS_CreateDefaultUserReference.restype = ViStatus
    dll.WFS_CreateDefaultUserReference.argtypes = [ViSession]

    dll.WFS_SaveUserRefFile.restype = ViStatus
    dll.WFS_SaveUserRefFile.argtypes = [ViSession]

    dll.WFS_LoadUserRefFile.restype = ViStatus
    dll.WFS_LoadUserRefFile.argtypes = [ViSession]

    dll.WFS_DoSphericalRef.restype = ViStatus
    dll.WFS_DoSphericalRef.argtypes = [ViSession]

    return dll

def np2c(x):
    return x.ctypes.data_as(ctypes.POINTER(c_int32))