import sys
import time
import logging
import socket

import numpy as np

import gxipy as gx

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
    dll.WFS_GetSpotfieldImageCopy.argtypes = [ViSession, ArrImg, POINTER(ViInt32), POINTER(ViInt32)]  # ViUInt8[]

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
    
    def __init__(self, mla_index:MlaRes, exp_time:float = 0.0, high_speed:bool = False, use_custom_ref:bool = False, pupil_diameter:float = 2.0):
        """
        mla_index: MlaRes
        exp_time: exposure time in ms, 0 means auto
        high_speed: enable high speed mode, only 512x512 resolution supported
        use_custom_ref: use custom reference file, if not, use default reference file
        pupil_diameter: pupil diameter in mm, default 2.0
        """
        assert mla_index in MlaRes, "mla_index must be one of MlaRes"
        
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
        if self._instrument_handle.value > 0:
            self.enable_high_speed = False

            self.__image_loop_counter = 0
            self._lib.WFS_close(self._instrument_handle)
            self._instrument_handle = c_ulong(0)

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
        self.__exit__()

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
                                spots_filed_img,
                                byref(c_int32(px)), byref(c_int32(py))
                                ):
            self.handle_error(err)
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

    def get_zernike(self, zernike_order=5, image_loop_counter: int = -1):
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
        assert 0.002 <= value <= 87, "exposure time must be in range [0.002, 87] ms"
        actual_exposure = c_double()
        self._lib.WFS_SetExposureTime(self._instrument_handle, c_double(value), byref(actual_exposure))
        print(f"actual exposure time is {actual_exposure.value} ms.")


    @property
    def pupil(self):
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

    #TODO: set high speed mode And get info enable in high speed mode
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

class CameraStreamManager:
    def __init__(self, cam_id:int=0, explosure_time:int=20, skip_sampling=True, log=logging.getLogger('galaxy camera driver')):
        self.device_manager = gx.DeviceManager()
        self.cam_id = cam_id
        self.explore_time = explosure_time
        self.skip_sampling = skip_sampling

        self.cam, self.__sn = None, None
        self.cam_width ,self.cam_height = 0, 0
        self.log = log

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cam:
            self.cam_width ,self.cam_height = 0, 0
            self.cam.stream_off()
            self.cam.close_device()
            self.cam, self.__sn = None, None

    def initialize(self):
        """
        初始化相机设备。

        此方法执行以下操作：
        1. 关闭之前打开的相机设备（如果有）。
        2. 更新设备列表并检查是否有足够的设备。
        3. 打开指定的相机设备。
        4. 设置相机的曝光时间、增益、像素格式、采样方式、偏移量、宽度和高度。
        5. 更新相机的属性并开启数据流。

        如果没有找到相机设备，将记录错误并抛出连接中止错误。

        参数:
            无

        返回:
            无
        """
        # 关闭之前打开的相机设备（如果有）
        self.__exit__(None, None, None)

        # 更新设备列表并获取设备信息列表
        _, dev_info_list = self.device_manager.update_device_list()
        # 检查设备列表长度是否小于等于指定的相机ID
        if len(dev_info_list) <= self.cam_id:
            self.log.error("No devices found.")
            raise ConnectionAbortedError("No cam devices found.")

        sn = dev_info_list[self.cam_id].get("sn")
        self.cam = self.device_manager.open_device_by_sn(sn)
        # 设置相机的曝光时间
        self.cam.ExposureTime.set(self.explore_time)
        # 设置相机的增益
        self.cam.Gain.set(0.0)
        # 设置相机的像素格式为MONO8
        self.cam.PixelFormat.set(gx.GxPixelFormatEntry.MONO8)
        if self.skip_sampling:
            # 设置相机的合并因子为2
            self.cam.BinningHorizontal.set(2)
            self.cam.BinningVertical.set(2)

        # 设置相机的水平偏移量为0
        self.cam.OffsetX.set(0)
        self.cam.OffsetY.set(0)
        # 设置相机的宽度为最大宽度
        self.cam.Width.set(self.cam.WidthMax.get())
        self.cam.Height.set(self.cam.HeightMax.get())

        self.__sn = sn
        self.__update_properties()
        self.cam.stream_on()

    def reset_explore_time(self, time:int):
        if time >= 20:
            self.explore_time = time
        else:
            self.explore_time = 20
            self.log.warning('explore time must >= 20. set to 20.')
        self.cam.ExposureTime.set(self.explore_time)
        return self.explore_time

    def reset_window(self, center:tuple[int, ...]|tuple[np.intp, ...], size:tuple[int,int]=(0,0)) -> tuple[tuple[int,int], tuple[int,int]]:
        """
        重置相机的窗口大小和位置，以确保图像的中心位于指定的位置。

        参数:
        size (Tuple[int]): 期望的窗口大小，格式为 (宽度, 高度)。
        center (Tuple[int]): 期望的窗口中心位置，格式为 (x坐标, y坐标)。

        返回:
        Tuple[int]: 新的窗口中心位置，格式为 (x坐标, y坐标)。
        """
        # 中心坐标大于0
        assert center[0]>0 and center[1]>0
        center = tuple(int(c) for c in center)
        if self.cam:
            self.cam.stream_off()
            # 如果未指定窗口大小，则使用相机的最大宽度和高度
            if size == (0, 0):
                size = (self.cam.WidthMax.get(), self.cam.HeightMax.get())
            width, height = size
            
            min_quatic = 16
            def reset_value(v):
                return int(v//min_quatic*min_quatic)
            width, height = reset_value(width), reset_value(height)
            # 计算窗口的偏移量，确保中心位置在指定位置
            x_offset, y_offset = center[0]-(width//2), center[1]-(height//2)
            x_offset, y_offset = reset_value(x_offset), reset_value(y_offset)
            assert x_offset>0 and y_offset>0
            self.cam.Width.set(width)
            self.cam.Height.set(height)
            self.cam.OffsetX.set(x_offset)
            # 设置相机的垂直偏移量，确保偏移量是4的倍数
            self.cam.OffsetY.set(y_offset)

            self.__update_properties()
            self.cam.stream_on()

            # 返回新的窗口中心位置
            return (width, height), (width//2, height//2)

    def get_numpy_image(self, n_sample=1) -> np.ndarray[np.uint8]:
        assert n_sample>0
        
        numpy_image = np.zeros((self.cam_height, self.cam_width))
        for _ in range(n_sample):
            while True:
                raw_image = self.cam.data_stream[0].get_image()
                if not raw_image:
                    continue
                
                numpy_image += raw_image.get_numpy_array()
                break
        avg_img = numpy_image/n_sample
        return avg_img.astype(np.uint8)

    def __update_properties(self):
        self.cam_width = self.cam.Width.get()
        self.cam_height = self.cam.Height.get()
        self.log.info(f"Open cam {self.__sn} success. width={self.cam_width}, height={self.cam_height}")
        self.xv, self.yv = self.__get_grid(self.cam_width, self.cam_height)

    @staticmethod
    def __get_grid(width, height):
        x = np.arange(0, width)
        y = np.arange(0, height)
        xv, yv = np.meshgrid(x, y)
        return xv, yv


class DMUdp:

    global HEAD_WITH_ECHO, HEAD, REG_IDS
    HEAD_WITH_ECHO = '10 01 2c'.split(' ')
    HEAD = '30 01 2c'.split(' ')
    REG_IDS = [0, 16384, 32768, 49152] # 0x00, 0x40, 0x80, 0xc0 to dec

    def __init__(self):
        self.ip = "192.168.6.10"
        self.port = 1001
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dm_num = 64

    @staticmethod
    def _num_hex(num:int):
        hex_16 = hex(int(num))[2:].zfill(4)
        return hex_16[:2]+' '+hex_16[2:]

    @staticmethod
    def _voltage_hex(num:float, registry:int):
        _num = int((num+500)/1000*4096)
        # _num = min(_num, 4095)
        # _num = max(_num, 820)
        _num += REG_IDS[registry%4]
        hex_16 = DMUdp._num_hex(_num)
        return hex_16
    
    def check_connection(self):
        # ping target ip
        try:
            socket.gethostbyname(self.ip)
            return True
        except socket.error:
            return False

    def _send(self, message):
        hex_message = bytes.fromhex(message)
        return self.sock.sendto(hex_message, (self.ip, self.port))
    
    def set_voltages(self, vs, with_echo=False):
        _head = HEAD_WITH_ECHO if with_echo else HEAD
        send_data = ' '.join(_head+[self._num_hex(self.dm_num)]+[self._voltage_hex(v,i) for i,v in enumerate(vs)])
        return self._send(send_data)
    
    def reset_all(self):
        vs = np.zeros(256)
        send_data = ' '.join(HEAD_WITH_ECHO+[self._num_hex(256)]+[self._voltage_hex(v,i) for i,v in enumerate(vs)])
        return self._send(send_data) & self._send("10 00 00 00 01 00 03")
    
    def set_hv(self, hv:bool):
        raise NotImplementedError()


class DMSdk:

    def __init__(self):
        self.dm_num = 64

        dll = cdll.LoadLibrary('Drv_UDPST/x64/Release/Drv_UDPST.dll')
        dll.SetVoltages.restype = c_bool
        dll.SetVoltages.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double, ndim=1, shape=(self.dm_num)), c_int32, c_int32]
        
        dll.SetVoltagesNoEcho.restype = c_bool
        dll.SetVoltagesNoEcho.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double, ndim=1, shape=(self.dm_num)), c_int32, c_int32]
        
        dll.SetHV.restype = c_bool
        dll.SetHV.argtypes = [c_bool, c_bool]

        self._dll = dll

    def set_voltages(self, vs:np.ndarray, with_echo=False):
        func = self._dll.SetVoltages if with_echo else self._dll.SetVoltagesNoEcho
        return func(vs, c_int32(0), c_int32(self.dm_num))
    
    def reset_all(self):
        return self._dll.ResetAll()
    
    def set_hv(self, hv:bool):
        return self._dll.SetHV(c_bool(hv), c_bool(True))
    
    def get_hv(self):
        hv_status = c_bool(False)
        if self._dll.GetHV(byref(hv_status)):
            return hv_status
        else:
            raise Exception("device connection error.")


class NlightDM:
    
    DM_Num = 64
    V_Min = -300
    V_Max = 499

    def __init__(
            self, 
            max_iter_diff=20, 
            max_neibor_diff=0, 
            warning_min = -180,
            warning_max = 200,
            keep_when_exit=True):
        assert max_iter_diff < 200
        assert max_neibor_diff < 200

        self.units_adj_mat = self._load_adj_txt()

        self.__last_v = np.zeros(self.DM_Num)
        self.max_iter_diff = max_iter_diff
        self.max_neibor_diff = max_neibor_diff
        self.warning_min = warning_min
        self.warning_max = warning_max

        self.c_driver = DMSdk()
        self.udp_driver = DMUdp()

        self.__keep_when_exit = keep_when_exit

    def __enter__(self):
        if not self.udp_driver.check_connection():
            raise Exception("device connection error.")

        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.__keep_when_exit:
            self.reset_all()
            self.set_hv(False)
            print("DM Turn off high voltages.")
        
        self.udp_driver.sock.close()

    @staticmethod
    def _load_adj_txt():
        return np.loadtxt('data/dm_adj.txt')
    
    def get_nerbors(self, unit_id):
        return np.where(self.units_adj_mat[unit_id, :] == 1)[0]
    
    def _reset_nerbors_voltage_in_range(self, unit_id, voltages, checked_mask):
        min, max = voltages[unit_id]-self.max_neibor_diff, voltages[unit_id]+self.max_neibor_diff
        for nerbor in self.get_nerbors(unit_id):
            if not checked_mask[unit_id, nerbor]:
                voltages[nerbor] = np.clip(voltages[nerbor], min, max)
                checked_mask[unit_id, nerbor] = checked_mask[nerbor, unit_id] = True
                self._reset_nerbors_voltage_in_range(nerbor, voltages, checked_mask)

    def initialize(self) -> None:
        self.set_hv(hv=True)

    def reset_all(self):
        self.send_voltages(np.zeros(self.DM_Num), 0.01)

        if (ret := self.c_driver.reset_all()) == 0:
            self.__last_v = np.zeros_like(self.__last_v)
        time.sleep(0.5)
        return ret

    def send_voltages(self, vs:np.ndarray, wait_time_s = 0.001):
        vs = np.clip(vs, self.V_Min, self.V_Max)
        __gap = vs - self.__last_v
        if self.max_iter_diff > 0:
            _direction = np.sign(__gap)
            _abs_gap = np.abs(__gap)
            while _abs_gap.any():
                _abs_gap = _abs_gap-self.max_iter_diff
                _abs_gap = np.where(_abs_gap<0, 0, _abs_gap)
                self.udp_driver.set_voltages(vs + _direction * _abs_gap)

        if np.max(vs) > self.warning_max:
            print(f"alert, votage higher than {self.warning_max}.", f"{np.argmax(vs)}={np.max(vs)}")
        if np.min(vs) < self.warning_min:
            print(f"alert, votage lower than {self.warning_min}.", f"{np.argmin(vs)}={np.min(vs)}")

        self.udp_driver.set_voltages(vs)
        self.__last_v = vs
        time.sleep(wait_time_s)
        return vs


    def set_hv(self, hv:bool = True):
        ret = self.c_driver.set_hv(hv)
        time.sleep(0.5)
        return ret

try:
    import bmc

    class BMCManager:

        def __init__(self, sn='17DW023#013', log=logging.getLogger('bmc dm driver')):

            self.sn = sn
            self.dm:bmc.BmcDm = None
            self.dm_num = 0
            self.log = log

        def initialize(self):
            self.dm = bmc.BmcDm()
            res = self.dm.open_dm(self.sn)
            if res:
                self.log.error(self.dm.error_string(res))
                return
            self.dm_num = self.dm.num_actuators()
            self.log.info(f'dm {self.sn} init. actuators count {self.dm_num}.')
        def __enter__(self):
            self.initialize()
            return self
        def __exit__(self, exc_type, exc_value, traceback):
            if self.dm:
                self.dm.close_dm()
            self.dm = None

        def set_data(self, data:np.ndarray):
            res = self.dm.send_data(data)
            if res:
                self.log.error(self.dm.error_string(res))
except Exception as e:
    print('use python 3.6 for bmc dm.')


if __name__ == '__main__':
    import numpy as np
    import math
    import tqdm
    import matplotlib.pyplot as plt
    def ture_off_dm():
        with NlightDM(keep_when_exit=False) as dm:
            v = np.zeros((dm.DM_Num,))
            dm.send_voltages(v)
            
    def load_last_v(file='last_v.npz', reset=False):
        with NlightDM(keep_when_exit=True, max_iter_diff=10) as dm:
            if file:
                load_v = np.load(file)['v'] if file.endswith('.npz') else np.loadtxt(file)
            init_V = np.zeros((dm.DM_Num,)) if reset else load_v
            dm.send_voltages(init_V, wait_time_s=0.01)
            
    def test_dm():
        import itertools
        with NlightDM(keep_when_exit=False) as dm:
            v = np.zeros((dm.DM_Num,))
            for phi in itertools.cycle(np.linspace(0, 2*np.pi, 10)):
                v[1] = math.sin(phi)*100
                dm.send_voltages(v, 0.1)
                print(v[1])

    def test_wfs():
        with WFSManager(MlaRes.Res512, exp_time=0.029) as wfs:
            # opt_exp_time, _ = wfs.optimize_exposure_time_and_gain()
            # if 0.001 < opt_exp_time < 87:
            #     wfs.exposure_time = opt_exp_time
            # else:
            #     print("no usable image. exit now..")
            #     exit()

            # print(f"optimize_pupil: {wfs.optimize_pupil()}")

            # for _ in range(1):
            #     wfs.take_image()
                
                
            #     spots_filed = wfs.get_spotfiled_image()
            #     plt.imshow(spots_filed)
            #     plt.show()
                
            #     x, y = wfs.get_spot_deviation()
            #     intensity, _ = wfs.get_spots_statics()
            #     wf, statics = wfs.get_rms()
            #     print(f"{statics=}")

            #     fig, ax = plt.subplots(2,2)
            #     ax[0,0].imshow(x)
            #     ax[0,0].set_title("spot deviation x")

            #     ax[0,1].imshow(y)
            #     ax[0,1].set_title("spot deviation y")
                
            #     ax[1,0].imshow(intensity)
            #     ax[1,0].set_title("spot intensity")
                
            #     ax[1,1].imshow(wf)
            #     ax[1,1].set_title("wavefront")
            #     plt.show()
                
            wfs.high_speed = True
            for _ in range(10):
                wfs.take_image()
                x,y = wfs.get_spot_deviation()
                print(y[0,:])
                # print(wfs.get_zernike(3))
                print(wfs.get_wavefront()[0][0,:])
    def test_cam(cam_id=0):
        with CameraStreamManager(cam_id, explosure_time=80) as cam:
            img = cam.get_numpy_image()
            center = np.unravel_index(np.argmax(img), img.shape)
            center = (center[1], center[0])
            print(f'{center=}')
            plt.imshow(img)
            plt.title(f'{center=} = {img[center[::-1]]=}')
            plt.show()
            
    # ture_off_dm()
    load_last_v(file='last_v-0.07.npz',reset=False)
    test_cam(0)
    # test_wfs()
    # 
    # test_dm()

    