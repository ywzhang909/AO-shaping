import sys
import time
import logging

import numpy as np

import ctypes
from ctypes import byref, c_double, create_string_buffer, c_bool, c_uint8, c_int16, c_int32, c_ulong, c_float, c_char,\
    c_char_p, cdll
from ctypes import POINTER
from enum import IntEnum

# WFS status bits
WFS_STATUS = {
    "CON" : 0x00000001,  # USB connection lost, set by driver
    "PTH" : 0x00000002,  # Power too high (cam saturated)
    "PTL" : 0x00000004,  # Power too low (low cam digits)
    "HAL" : 0x00000008,  # High ambient light
    "SCL" : 0x00000010,  # Spot contrast too low
    "ZFL" : 0x00000020,  # Zernike fit failed because of not enough detected spots
    "ZFH" : 0x00000040,  # Zernike fit failed because of too much detected spots
    "ATR" : 0x00000080,  # Camera is still awaiting a trigger
    "CFG" : 0x00000100,  # Camera is configured, ready to use
    "PUD" : 0x00000200,  # Pupil is defined
    "SPC" : 0x00000400,  # No. of spots or pupil or aoi has been changed
    "RDA" : 0x00000800,  # Reconstructed spot deviations available
    "URF" : 0x00001000,  # User reference data available
    "HSP" : 0x00002000,  # Camera is in Highspeed Mode
    "MIS" : 0x00004000,  # Mismatched centroids in Highspeed Mode
    "LOS" : 0x00008000,  # low number of detected spots, warning: reduced Zernike accuracy
    "FIL" : 0x00010000  # pupil is badly filled with spots, warning: reduced Zernike accuracy
}


# MAX_SPOTS is actually a constrained by the library version
# see WFS.h for the actual value
MAX_SPOTS = [80, 80]
EXP_TIME_LOW = 0.002
EXP_TIME_HIGH = 86

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
ArrImg = np.ctypeslib.ndpointer(dtype=np.uint8, shape=(512,512))
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
    dll.WFS_GetInstrumentInfo.argtypes = [ViSession, ViChar256, ViChar256, ViChar256, ViChar256]
    # ViStatus __fastcall WFS_GetInstrumentListInfo(ViSession instrumentHandle, ViInt32 instrumentListIndex, ViInt32 *deviceID, ViInt32 *inUse, ViChar *instrumentName, ViChar *instrumentSN, ViChar *resourceName)
    dll.WFS_GetInstrumentListInfo.restype = ViStatus
    dll.WFS_GetInstrumentListInfo.argtypes = [ViSession, ViInt32, POINTER(ViInt32), POINTER(ViInt32), ViRsrc, ViRsrc, ViRsrc]

    dll.WFS_ConfigureCam.restype = ViStatus
    dll.WFS_ConfigureCam.argtypes = [ViSession, ViInt32, ViInt32, POINTER(ViInt32), POINTER(ViInt32)]

    dll.WFS_SetHighspeedMode.restype = ViStatus
    dll.WFS_SetHighspeedMode.argtypes = [ViSession, ViInt32, ViInt32, ViInt32, ViInt32]

    # dll.WFS_GetHighspeedWindows.restype = ViStatus
    # dll.WFS_GetHighspeedWindows.argtypes = [ViSession, POINTER(ViInt32), POINTER(ViInt32), POINTER(ViInt32), POINTER(ViInt32), ViInt32, ViInt32]

    dll.WFS_CheckHighspeedCentroids.restype = ViStatus
    dll.WFS_CheckHighspeedCentroids.argtypes = [ViSession]

    dll.WFS_GetExposureTimeRange.restype = ViStatus
    dll.WFS_GetExposureTimeRange.argtypes = [ViSession, POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64)]

    dll.WFS_SetExposureTime.restype = ViStatus
    dll.WFS_SetExposureTime.argtypes = [ViSession, ViReal64, POINTER(ViReal64)]

    dll.WFS_GetExposureTime.restype = ViStatus
    dll.WFS_GetExposureTime.argtypes = [ViSession, POINTER(ViReal64)]

    dll.WFS_GetMasterGainRange.restype = ViStatus
    dll.WFS_GetMasterGainRange.argtypes = [ViSession, POINTER(ViReal64), POINTER(ViReal64)]

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
    dll.WFS_GetMlaData.argtypes = [ViSession, ViInt32, ViChar256, POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64)]

    dll.WFS_GetMlaData2.restype = ViStatus
    dll.WFS_GetMlaData2.argtypes = [ViSession, ViInt32, ViChar256, POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64)]

    dll.WFS_SelectMla.restype = ViStatus
    dll.WFS_SelectMla.argtypes = [ViSession, ViInt32]

    # WFS_SetAoi and WFS_SetAoi are undocumented and thus left out

    dll.WFS_SetPupil.restype = ViStatus
    dll.WFS_SetPupil.argtypes = [ViSession, ViReal64, ViReal64, ViReal64, ViReal64]

    dll.WFS_GetPupil.restype = ViStatus
    dll.WFS_GetPupil.argtypes = [ViSession, POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64)]

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
    dll.WFS_TakeSpotfieldImageAutoExpos.argtypes = [ViSession, POINTER(ViReal64), POINTER(ViReal64)]

    # WFS_GetSpotfieldImage left out

    dll.WFS_GetSpotfieldImageCopy.restype = ViStatus
    dll.WFS_GetSpotfieldImageCopy.argtypes = [ViSession, ctypes.POINTER(c_uint8), POINTER(ViInt32), POINTER(ViInt32)]  # ViUInt8[]

    dll.WFS_AverageImage.restype = ViStatus
    dll.WFS_AverageImage.argtypes = [ViSession, ViInt32, POINTER(ViInt32)]

    dll.WFS_AverageImageRolling.restype = ViStatus
    dll.WFS_AverageImageRolling.argtypes = [ViSession, ViInt32, ViInt32]

    dll.WFS_CutImageNoiseFloor.restype = ViStatus
    dll.WFS_CutImageNoiseFloor.argtypes = [ViSession, ViInt32]

    dll.WFS_CalcImageMinMax.restype = ViStatus
    dll.WFS_CalcImageMinMax.argtypes = [ViSession, POINTER(ViInt32), POINTER(ViInt32), POINTER(ViReal64)]

    dll.WFS_CalcMeanRmsNoise.restype = ViStatus
    dll.WFS_CalcMeanRmsNoise.argtypes = [ViSession, POINTER(ViReal64), POINTER(ViReal64)]

    dll.WFS_GetLine.restype = ViStatus
    dll.WFS_GetLine.argtypes = [ViSession, ViInt32, c_float] # float[]

    dll.WFS_GetLineView.restype = ViStatus
    dll.WFS_GetLineView.argtypes = [ViSession, c_float, c_float] # float[]

    dll.WFS_CalcBeamCentroidDia.restype = ViStatus
    dll.WFS_CalcBeamCentroidDia.argtypes = [ViSession, POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64)]

    dll.WFS_CalcSpotsCentrDiaIntens.restype = ViStatus
    dll.WFS_CalcSpotsCentrDiaIntens.argtypes = [ViSession, ViInt32, ViInt32]

    dll.WFS_GetSpotCentroids.restype = ViStatus
    dll.WFS_GetSpotCentroids.argtypes = [ViSession, ArrFloat, ArrFloat] # float[]

    dll.WFS_GetSpotDiameters.restype = ViStatus
    dll.WFS_GetSpotDiameters.argtypes = [ViSession, c_float, c_float] # float[]

    dll.WFS_GetSpotDiaStatistics.restype = ViStatus
    dll.WFS_GetSpotDiaStatistics.argtypes = [ViSession, POINTER(ViInt32), POINTER(ViInt32), POINTER(ViInt32)]

    dll.WFS_GetSpotIntensities.restype = ViStatus
    dll.WFS_GetSpotIntensities.argtypes = [ViSession, ArrFloat] # float[]

    dll.WFS_CalcSpotToReferenceDeviations.restype = ViStatus
    dll.WFS_CalcSpotToReferenceDeviations.argtypes = [ViSession, ViInt32]

    dll.WFS_GetSpotReferencePositions.restype = ViStatus
    dll.WFS_GetSpotReferencePositions.argtypes = [ViSession, c_float, c_float] # float[]

    dll.WFS_GetSpotDeviations.restype = ViStatus
    dll.WFS_GetSpotDeviations.argtypes = [ViSession, ArrFloat, ArrFloat] # float[]

    # dll.WFS_ZernikeLsf.restype = ViStatus
    # dll.WFS_ZernikeLsf.argtypes = [ViSession, POINTER(ViInt32), c_float, c_float, POINTER(ViReal64)] # float[]

    dll.WFS_CalcFourierOptometric.restype = ViStatus
    dll.WFS_CalcFourierOptometric.argtypes = [ViSession, ViInt32, ViInt32, POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64)]

    dll.WFS_CalcReconstrDeviations.restype = ViStatus
    dll.WFS_CalcReconstrDeviations.argtypes = [ViSession, ViInt32, ViInt32, ViInt32, POINTER(ViReal64), POINTER(ViReal64)] # ViInt32[]

    dll.WFS_CalcWavefront.restype = ViStatus
    dll.WFS_CalcWavefront.argtypes = [ViSession, ViInt32, ViInt32, ArrFloat] # float[]

    dll.WFS_CalcWavefrontStatistics.restype = ViStatus
    dll.WFS_CalcWavefrontStatistics.argtypes = [ViSession, POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64), POINTER(ViReal64)]

    # Utility Functions
    dll.WFS_self_test.restype = ViStatus
    dll.WFS_self_test.argtypes = [ViSession, ViInt16, c_char_p] # ViChar[]

    dll.WFS_reset.restype = ViStatus
    dll.WFS_reset.argtypes = [ViSession]

    dll.WFS_revision_query.restype = ViStatus
    dll.WFS_revision_query.argtypes = [ViSession, c_char_p, c_char_p] # ViChar[]

    dll.WFS_error_query.restype = ViStatus
    dll.WFS_error_query.argtypes = [ViSession, POINTER(ViInt32), c_char_p] # ViChar[]

    dll.WFS_error_message.restype = ViStatus
    dll.WFS_error_message.argtypes = [ViSession, ViStatus, POINTER(ViChar256)]

    dll.WFS_GetInstrumentListLen.restype = ViStatus
    dll.WFS_GetInstrumentListLen.argtypes = [ViSession, POINTER(ViInt32)]

    dll.WFS_GetInstrumentListInfo.restype = ViStatus
    dll.WFS_GetInstrumentListInfo.argtypes = [ViSession, ViInt32, POINTER(ViInt32), POINTER(ViInt32), ViChar256, ViChar256, ViRsrc] # ViChar[]

    dll.WFS_GetXYScale.restype = ViStatus
    dll.WFS_GetXYScale.argtypes = [ViSession, c_float, c_float] # float[]

    dll.WFS_ConvertWavefrontWaves.restype = ViStatus
    dll.WFS_ConvertWavefrontWaves.argtypes = [ViSession, ViReal64, ViReal32, ViReal32] # ViReal[]

    dll.WFS_Flip2DArray.restype = ViStatus
    dll.WFS_Flip2DArray.argtypes = [ViSession, ViReal32, ViReal32] # ViReal32

    # Calibration Functions
    dll.WFS_SetSpotsToUserReference.restype = ViStatus
    dll.WFS_SetSpotsToUserReference.argtypes = [ViSession]

    dll.WFS_SetCalcSpotsToUserReference.restype = ViStatus
    dll.WFS_SetCalcSpotsToUserReference.argtypes = [ViSession, ViInt32, c_float, c_float] # float[]

    dll.WFS_CreateDefaultUserReference.restype = ViStatus
    dll.WFS_CreateDefaultUserReference.argtypes = [ViSession]

    dll.WFS_SaveUserRefFile.restype = ViStatus
    dll.WFS_SaveUserRefFile.argtypes = [ViSession]

    dll.WFS_LoadUserRefFile.restype = ViStatus
    dll.WFS_LoadUserRefFile.argtypes = [ViSession]

    dll.WFS_DoSphericalRef.restype = ViStatus
    dll.WFS_DoSphericalRef.argtypes = [ViSession]
    
    return dll

#define  CAM_RES_1280                  (0) // 1280x1024
#define  CAM_RES_1024                  (1) // 1024x1024
#define  CAM_RES_768                   (2) // 768x768
#define  CAM_RES_512                   (3) // 512x512
#define  CAM_RES_320                   (4) // 320x320 smallest!
class MlaRes(IntEnum):
    Res1280 = 0
    Res1024 = 1
    Res768 = 2
    Res512 = 3
    Res320 = 4
    
Mla_pix = {
    MlaRes.Res1280: (1280,1024),
    MlaRes.Res1024: (1024,1024),
    MlaRes.Res768: (768,768),
    MlaRes.Res512: (512,512),
    MlaRes.Res320: (320,320),
}

def np2c(x):
    return x.ctypes.data_as(ctypes.POINTER(c_int32))


class WFSManager:
    """ Wavefront Sensor Manager
    """
    
    def __init__(self, mla_index:MlaRes = MlaRes.Res768, exp_time:float = 0.0, high_speed:bool = False, use_custom_ref:bool = False, pupil_diameter:float = 2.0):
        """
        mla_index: MlaRes
        exp_time: exposure time in ms, 0 means auto
        high_speed: enable high speed mode, only 512x512 resolution supported
        use_custom_ref: use custom reference file, if not, use default reference file
        pupil_diameter: pupil diameter in mm, default 2.0
        """
        assert mla_index in MlaRes, "mla_index must be one of MlaRes"
        assert exp_time==0.0 or EXP_TIME_LOW <= exp_time <= EXP_TIME_HIGH, f"exp_time must be in [{EXP_TIME_LOW},{EXP_TIME_HIGH}], now is {exp_time}"
        
        self._lib = load_dll()
        self.device_id = c_int32()
        self._instrument_handle = c_ulong(0)

        self.use_custom_ref = use_custom_ref
        self.mla_index = mla_index
        self.image_pix = Mla_pix[mla_index]
        self.num_spots_x, self.num_spots_y = 0, 0
        self.c_x, self.c_y = 0.0, 0.0
        self.d_x, self.d_y = pupil_diameter, pupil_diameter

        self._explosure_time = exp_time
        self._gain = 1.0
        self.enable_high_speed = high_speed
        if self.enable_high_speed:
            print("high speed mode can only use auto explore time!")
        self.__image_loop_counter = 0

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def initialize(self):
        device_in_use = ViInt32()
        device_name = create_string_buffer(256)
        serial_number = create_string_buffer(256)
        resource_name = create_string_buffer(256)
        self._lib.WFS_GetInstrumentListInfo(VI_NULL(), ViInt32(0), byref(self.device_id), byref(device_in_use), device_name,
                                        serial_number, resource_name)

        #check if WFS is in use, if not, connect to device
        assert not device_in_use, "Wavefront sensor currently in use.... closing program"

        self._lib.WFS_init(resource_name, c_bool(False), c_bool(True), byref(self._instrument_handle))
        print(f"Connected to {device_name.value} with Serial Number {serial_number.value}")

        self.select_mla(self.mla_index)
        self.set_ref_plane(self.use_custom_ref)
        self.pupil = (self.c_x, self.c_y, self.d_x, self.d_y)
        if self._explosure_time <= 0:
            self._explosure_time,_ = self.optimize_exposure_time_and_gain()
        self.exposure_time = self._explosure_time
        self.high_speed = self.enable_high_speed
        self.pupil = self.pupil if (self.d_x>0 and self.d_y>0) else self.optimize_pupil()
        
    def close(self):
        if self._instrument_handle.value > 0:
            self.enable_high_speed = False

            self.__image_loop_counter = 0
            self._lib.WFS_close(self._instrument_handle)
            self._instrument_handle = c_ulong(0)

    def handle_error(self, err, no_raise=False):
        info = create_string_buffer(256)
        errorCode = ViStatus(err)
        self._lib.WFS_error_message(self._instrument_handle, errorCode, byref(info))
        print("error:", str(info.value))
        if no_raise:
            print(info.value)
        else:
            raise Exception(info.value)

    def select_mla(self, mla_index:MlaRes):
        self._lib.WFS_SelectMla(self._instrument_handle, 0)
        num_spots_x = c_int32()
        num_spots_y = c_int32()
        self._lib.WFS_ConfigureCam(
            self._instrument_handle, c_int32(0), c_int32(mla_index.value), byref(num_spots_x), byref(num_spots_y))
        self.mla_index = mla_index
        self.image_pix = Mla_pix[mla_index]
        self.num_spots_x, self.num_spots_y = num_spots_x.value, num_spots_y.value
        print(f"Number of detectable spots in X: {num_spots_x.value} \n"+
              f"Number of detectable spots in Y: {num_spots_y.value}")

    def set_ref_plane(self, custom:bool):
        _select = 1 if custom else 0
        if err:= self._lib.WFS_SetReferencePlane(self._instrument_handle, c_int32(_select)):
            self.handle_error(err)

        elif custom:
            if err:= self._lib.WFS_LoadUserRefFile(self._instrument_handle):
                self.handle_error(err)
            else:
                self.use_custom_ref = True
                return
        
        self.use_custom_ref = False


    def optimize_pupil(self):
        '''
        This function help to optimize pupil.
        Returns:
            tuple[float, float, float, float]: beam centroid x, beam centroid y, beam diameter x, beam diameter y
        '''
        assert not self.enable_high_speed, "turn off high speed mode first"
        self._lib.WFS_CalcSpotsCentrDiaIntens(self._instrument_handle, c_int32(1), c_int32(1))
        beam_centroid_x = c_double()
        beam_centroid_y = c_double()
        beam_diameter_x = c_double()
        beam_diameter_y = c_double()
        self._lib.WFS_CalcBeamCentroidDia(
            self._instrument_handle,
            byref(beam_centroid_x), byref(beam_centroid_y), byref(beam_diameter_x), byref(beam_diameter_y))
        return beam_centroid_x.value, beam_centroid_y.value, beam_diameter_x.value, beam_diameter_y.value

    def take_image(self, n_sample=10):
        if self._explosure_time > 0:
            if err := self._lib.WFS_TakeSpotfieldImage(self._instrument_handle):
                self.handle_error(err)
            else:
                self.__image_loop_counter = (self.__image_loop_counter + 1) % (sys.maxsize - 1)
        else:
            actual_exposure = c_double()
            actual_gain = c_double()
            for _ in range(n_sample):
                self._lib.WFS_TakeSpotfieldImageAutoExpos(
                    self._instrument_handle, byref(actual_exposure), byref(actual_gain))
            self.__image_loop_counter = (self.__image_loop_counter + 1) % (sys.maxsize - 1)
        if res := self._lib.WFS_CalcSpotToReferenceDeviations(self._instrument_handle, c_int32(1)):
                self.handle_error(res)
                
    def get_spotfiled_image(self, image_loop_counter: int = -1):
        px, py = self.image_pix
        spots_filed_img = np.zeros((px,py), np.uint8)
        if err:= self._lib.WFS_GetSpotfieldImageCopy(self._instrument_handle,
                                spots_filed_img.ctypes.data_as(ctypes.POINTER(c_uint8)),
                                byref(c_int32(px)), byref(c_int32(py))
                                ):
            raise RuntimeError(self.handle_error(err))
        else:
            return spots_filed_img
        
    def get_spots_statics(self, image_loop_counter: int = -1):
        assert not self.enable_high_speed, "turn off high speed mode first"

        if err := self._lib.WFS_CalcSpotsCentrDiaIntens(self._instrument_handle, ViInt32(0), ViInt32(1)):
            self.handle_error(err)
        else:
            spots_intensities = np.zeros(MAX_SPOTS, dtype= np.float32)
            spots_center_x, spots_center_y = spots_intensities.copy(), spots_intensities.copy()
            # spots_diameter_x, spots_diameter_y = spots_intensities.copy(), spots_intensities.copy()
            self._lib.WFS_GetSpotIntensities(
                self._instrument_handle, spots_intensities)
            self._lib.WFS_GetSpotCentroids(self._instrument_handle,
                spots_center_x, spots_center_y)
            # self._lib.WFS_GetSpotDiameters(self._instrument_handle,
            #     np2c(spots_diameter_x), np2c(spots_diameter_y))
        return spots_intensities[:self.num_spots_x, :self.num_spots_y], (spots_center_x[:self.num_spots_x, :self.num_spots_y], spots_center_y[:self.num_spots_x, :self.num_spots_y])

    def get_wavefront(self, image_loop_counter: int = -1):
        '''
        This function help to get wavefront.
        Args:
            image_loop_counter (int, optional): Image loop counter. Defaults to -1.
        Returns:
            tuple[np.ndarray, dict]: wavefront, wavefront statistics
        '''
        adaptive_pupil = 0 if (self.d_x and self.d_y) else 1
        wavefront = np.zeros(MAX_SPOTS, dtype=c_float)
        if err := self._lib.WFS_CalcWavefront(
            self._instrument_handle, ViInt32(0), ViInt32(adaptive_pupil), wavefront):
            self.handle_error(err)
        else:
            min, max, diff, mean = c_double(), c_double(), c_double(), c_double()
            rms, wighted_rms = c_double(), c_double()
            self._lib.WFS_CalcWavefrontStatistics(
                self._instrument_handle, byref(min), byref(max), byref(diff), byref(mean),
                byref(rms), byref(wighted_rms)
            )

        wavefront = wavefront[:self.num_spots_x, :self.num_spots_y]
        # wavefront = np.where(wavefront==np.nan, 0, wavefront)
        return wavefront, {"min":min.value, "max":max.value, "diff":diff.value, "mean":mean.value, "rms":rms.value, "wighted_rms":wighted_rms.value}

    def get_zernike(self, zernike_order=10, image_loop_counter: int = -1):
        '''
        This function help to get zernike coefficients.
        Args:
            zernike_order (int, optional): Zernike order. Defaults to 10.
        Returns:
            np.ndarray: zernike coefficients
        '''
        assert zernike_order <= 10, "zernike order must be less than or equal to 10"
        roc_mm = c_double()
        coeff_num =  (zernike_order + 1) * (zernike_order + 2) // 2 + 1
        zernike_order = c_int32(zernike_order)
        zernike_um = np.zeros((coeff_num,), c_float)
        zernike_orders_rms_um = np.zeros((11,), c_float)
        if err:= self._lib.WFS_ZernikeLsf(self._instrument_handle, byref(zernike_order),
                                    zernike_um.ctypes.data_as(ctypes.POINTER(c_float)),
                                    zernike_orders_rms_um.ctypes.data_as(ctypes.POINTER(c_float)),
                                    byref(roc_mm)
                                    ):
            self.handle_error(err)
        else:
            
            return zernike_um

    def get_spot_deviation(self, cancel_tile:bool = False):
        '''
        This function help to get spot deviation.
        Args:
            cancel_tile (bool, optional): Whether to cancel tile. Defaults to False.
        Returns:
            tuple[np.ndarray, np.ndarray]: spot deviation x, spot deviation y
        '''
        spots_deviation_x = np.zeros(MAX_SPOTS, dtype= np.float32)
        spots_deviation_y = np.zeros(MAX_SPOTS, dtype= np.float32)
        # if err:= self._lib.WFS_CalcSpotsCentrDiaIntens(self._instrument_handle, c_int32(1), c_int32(1)):
        #     self.handle_error(err)
        
        if (res := self._lib.WFS_CalcSpotToReferenceDeviations(
            self._instrument_handle, c_int32(1 if cancel_tile else 0))) == 0:
            if err:= self._lib.WFS_GetSpotDeviations(self._instrument_handle, spots_deviation_x, spots_deviation_y):
                self.handle_error(err)
        else:
            self.handle_error(res)

        return spots_deviation_x[:self.num_spots_x, :self.num_spots_y], spots_deviation_y[:self.num_spots_x, :self.num_spots_y]


    def optimize_exposure_time_and_gain(self) -> tuple[float, float]:
        '''
        This function help to find reasonable exposure time, will NOT change it.
        
        Args:
        
        Returns:
            tuple[float, float]: exposure time, gain
        '''
        lib, instrument_handle = self._lib, self._instrument_handle
        #Take a series of images until one is usable. Check the device status after each image to determine usability
        actual_exposure = c_double()
        actual_gain = c_double()
        device_status = c_int32()
        for i in range(10):
            lib.WFS_TakeSpotfieldImageAutoExpos(instrument_handle, byref(actual_exposure), byref(actual_gain))
            lib.WFS_GetStatus(instrument_handle, byref(device_status))
            if device_status.value & 0x00000002:
                print("Power too high")
            elif device_status.value & 0x00000004:
                print("Power too low")
            elif device_status.value & 0x00000008:
                print("High ambient light")
            else:
                print(f"Image is usable at {actual_exposure.value} ms.... breaking loop")
                break
        return actual_exposure.value, actual_gain.value

    @property
    def exposure_time(self):
        actual_exposure = c_double()
        self._lib.WFS_GetExposureTime(self._instrument_handle, actual_exposure)
        return actual_exposure

    @exposure_time.setter
    def exposure_time(self, value: float):
        assert EXP_TIME_LOW <= value <= EXP_TIME_HIGH, f"exposure time must be in range [{EXP_TIME_LOW}, {EXP_TIME_HIGH}] ms"
        actual_exposure = c_double()
        self._lib.WFS_SetExposureTime(self._instrument_handle, c_double(value), byref(actual_exposure))
        print(f"actual exposure time is {actual_exposure.value} ms.")


    @property
    def pupil(self):
        '''
        This function help to get pupil.
        Returns:
            tuple[float, float, float, float]: beam centroid x, beam centroid y, beam diameter x, beam diameter y
        '''
        beam_centroid_x = c_double()
        beam_centroid_y = c_double()
        beam_diameter_x = c_double()
        beam_diameter_y = c_double()
        self._lib.WFS_GetPupil(
            self._instrument_handle,
            byref(beam_centroid_x), byref(beam_centroid_y), byref(beam_diameter_x), byref(beam_diameter_y))
        self.c_x, self.c_y = beam_centroid_x.value, beam_centroid_y.value
        self.d_x, self.d_y = beam_diameter_x.value, beam_diameter_y.value
        return self.c_x, self.c_y, self.d_x, self.d_y

    @pupil.setter
    def pupil(self, center_and_diameter:tuple):
        c_x, c_y, d_x, d_y = center_and_diameter
        self._lib.WFS_SetPupil(
            self._instrument_handle,
            c_double(c_x), c_double(c_y), c_double(d_x), c_double(d_y))
        self.c_x, self.c_y, self.d_x, self.d_y = c_x, c_y, d_x, d_y

    @property
    def high_speed(self):
        enable_high_speed = self._lib.WFS_CheckHighspeedCentroids(self._instrument_handle).value
        return enable_high_speed.value

    @high_speed.setter
    def high_speed(self, enable: bool):
        '''
        instrumentHandle	ViSession	This parameter accepts the Instrument Handle returned by the Init function to select the desired instrument driver session.
        highspeedMode	ViInt32	This parameter determines if the camera's Highspeed Mode is switched on or off.
        adaptCentroids	ViInt32	When Highspeed Mode is selected, this parameter determines if the centroid positions measured in Normal Mode should be used to adapt the spot search windows for Highspeed Mode.
        Otherwise, a rigid grid based on reference spot positions is used in Highspeed Mode.
        substractOffset	ViInt32	This parameter defines an offset level for Highspeed Mode only. All camera pixels will be subtracted by this level before the centroids are being calculated, which increases accuracy.
        Valid range: 0 ... 255
        Note: The offset is only valid in Highspeed Mode and must not set too high to clear the spots within the camera image!
        allowAutoExposure	ViInt32	When Highspeed Mode is selected, this parameter determines if the camera should also calculate the image saturation in order enable the auto exposure feature using function WFS_TakeSpotfieldImageAutoExpos() instead of WFS_TakeSpotfieldImage().
        This option leads to a somewhat reduced measurement speed when enabled.
                '''
        self.enable_high_speed = False
        if enable:
            self.optimize_exposure_time_and_gain()
            self._lib.WFS_CalcSpotsCentrDiaIntens(self._instrument_handle, c_int32(1), c_int32(1))
            self.pupil = (self.c_x, self.c_y, self.d_x, self.d_y)

        if (res := self._lib.WFS_SetHighspeedMode(self._instrument_handle,
                                       c_int32(1 if enable else 0), c_int32(1), c_int32(1), c_int32(1))) != 0:
            self.handle_error(res, True)
        else:
            self.enable_high_speed = enable
            if self.enable_high_speed:
                windowCountX = ViInt32() # This parameter returns the number of spot windows in X direction.
                windowCountY = ViInt32() # This parameter returns the number of spot windows in Y direction.
                windowSizeX = ViInt32() # This parameter returns the size in pixels of spot windows in X direction.
                windowSizeY = ViInt32() # This parameter returns the size in pixels of spot windows in Y direction.
                windowStartposX = np.zeros(self.num_spots_x, dtype=np.int32) # This parameter returns a one-dimensional array containing the start positions in pixels for spot windows in X direction.
                windowStartposY = np.zeros(self.num_spots_y, dtype=np.int32) # This parameter returns a one-dimensional array containing the start positions in pixels for spot windows in Y direction.

                self._lib.WFS_GetHighspeedWindows(
                    self._instrument_handle,
                    byref(windowCountX),
                    byref(windowCountY),
                    byref(windowSizeX),
                    byref(windowSizeY),
                    np2c(windowStartposX),
                    np2c(windowStartposY))

                self.hs_window_count_x = windowCountX.value
                self.hs_window_count_y = windowCountY.value
                self.hs_window_size_x = windowSizeX.value
                self.hs_window_size_y = windowSizeY.value
                self.hs_window_startpos_x = windowStartposX
                self.hs_window_startpos_y = windowStartposY

        print("high speed mode is " + "on" if enable else "off")



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def test_wfs():
        with WFSManager(MlaRes.Res512, exp_time=0.029) as wfs:
            opt_exp_time, _ = wfs.optimize_exposure_time_and_gain()
            if 0.001 < opt_exp_time < 87:
                wfs.exposure_time = opt_exp_time
            else:
                print("no usable image. exit now..")
                exit()

            print(f"optimize_pupil: {wfs.optimize_pupil()}")

            for _ in range(1):
                wfs.take_image()
                
                spots_filed = wfs.get_spotfiled_image()
                plt.imshow(spots_filed)
                plt.show()
                
                x, y = wfs.get_spot_deviation()
                intensity, _ = wfs.get_spots_statics()
                wf, statics = wfs.get_wavefront()
                print(f"{statics=}")

                fig, ax = plt.subplots(2,2)
                ax[0,0].imshow(x)
                ax[0,0].set_title("spot deviation x")

                ax[0,1].imshow(y)
                ax[0,1].set_title("spot deviation y")
                
                ax[1,0].imshow(intensity)
                ax[1,0].set_title("spot intensity")
                
                ax[1,1].imshow(wf)
                ax[1,1].set_title("wavefront")
                plt.show()
                
            wfs.high_speed = True
            for _ in range(10):
                wfs.take_image()
                x,y = wfs.get_spot_deviation()
                print(y[0,:])
                # print(wfs.get_zernike(3))
                print(wfs.get_wavefront()[0][0,:])
                
    def test_rms():
        rms_hist = []
        with WFSManager(MlaRes.Res768) as wfs:
            wfs.high_speed = True
            for _ in range(100):
                wfs.take_image(n_sample=10)
                
                # dx, dy = wfs.get_spot_deviation()
                # rms = np.sqrt(np.nanmean(dx**2+dy**2))
                zernike_coeff = wfs.get_zernike(10)
                print(zernike_coeff)
                rms_hist.append(np.mean(np.sqrt(np.sum(zernike_coeff**2))))
        return rms_hist
    # test_wfs()

    rms_hist = test_rms()
    plt.plot(rms_hist)
    plt.show()
    