import numpy as np
import time
import socket
from ao_shaping.drivers.dm.base import DM

from ctypes import byref, c_bool, c_int32, cdll
import findlibs

DM_NUM = 64

class NLight(DM):
    DM_Num: int = DM_NUM
    v_min, v_max = -300, 499

    def __init__(self, max_iter_diff=20, max_neibor_diff=0, keep_when_exit=True):
        assert max_iter_diff <= 200
        assert max_neibor_diff <= 300

        self.units_adj_mat = self._load_adj_txt()

        self.__last_v = np.zeros(self.DM_Num)
        self.max_iter_diff = max_iter_diff
        self.max_neibor_diff = max_neibor_diff

        self.c_driver = DMSdk()
        self.udp_driver = DMUdp()

        self.__keep_when_exit = keep_when_exit

    def open(self) -> None:
        """Open connection to DM and initialize"""
        self.initialize()

    def close(self) -> None:
        """Close connection to DM and clean up"""
        if not self.__keep_when_exit:
            self.reset_all()
            self.set_hv(False)
            print("DM Turn off high voltages.")
        self.udp_driver.sock.close()

    def transform(self, cmd:np.ndarray) -> np.ndarray:
        """Transform command to DM actuators"""
        cmd = np.clip(cmd, -1, 1)
        return (cmd + 1) * (self.v_max - self.v_min) / 2 + self.v_min

    def send(self, cmd):
        """Send command to DM - accepts voltage array"""
        if isinstance(cmd, np.ndarray):
            return self.send_voltages(cmd)
        raise ValueError("Unsupported command type. Expected numpy array of voltages.")

    def get_actuator_positions(self):
        """Get positions of DM actuators"""
        # Implementation would depend on specific hardware capabilities
        pass

    def initialize(self) -> None:
        self.set_hv(hv=True)

    def reset_all(self):
        self.send_voltages(np.zeros(self.DM_Num), 0.01)

        if (ret := self.c_driver.reset_all()) == 0:
            self.__last_v = np.zeros_like(self.__last_v)
        time.sleep(0.5)
        return ret

    def send_voltages(self, vs: np.ndarray, wait_time_s=0.001):
        vs = np.clip(vs, -300, 499)
        __gap = vs - self.__last_v
        if self.max_iter_diff > 0:
            _direction = np.sign(__gap)
            _abs_gap = np.abs(__gap)
            while _abs_gap.any():
                _abs_gap = np.clip(_abs_gap - self.max_iter_diff, 0, 499)
                self.udp_driver.set_voltages(vs + _direction * _abs_gap)
        self.udp_driver.set_voltages(vs)
        self.__last_v = vs
        time.sleep(wait_time_s)
        return self.__last_v

    def set_hv(self, hv: bool = True):
        ret = self.c_driver.set_hv(hv)
        time.sleep(0.5)
        return ret

    @staticmethod
    def _load_adj_txt():
        return np.loadtxt('data/dm_adj.txt')

    def get_neighbors(self, unit_id):
        return np.where(self.units_adj_mat[unit_id, :] == 1)[0]

    def _reset_nerbors_voltage_in_range(self, unit_id, voltages, checked_mask=None):
        if checked_mask is None:
            checked_mask = np.zeros_like(self.units_adj_mat, dtype=bool)
        min_v, max_v = voltages[unit_id]-self.max_neibor_diff, voltages[unit_id]+self.max_neibor_diff
        for nerbor in self.get_neighbors(unit_id):
            if not checked_mask[unit_id, nerbor]:
                voltages[nerbor] = np.clip(voltages[nerbor], min_v, max_v)
                checked_mask[unit_id, nerbor] = checked_mask[nerbor, unit_id] = True
                self._reset_nerbors_voltage_in_range(nerbor, voltages, checked_mask)
        return voltages

class DMUdp:

    global HEAD_WITH_ECHO, HEAD, REG_IDS
    HEAD_WITH_ECHO = '10 01 2c'.split(' ')
    HEAD = '30 01 2c'.split(' ')
    REG_IDS = [0, 16384, 32768, 49152] # 0x00, 0x40, 0x80, 0xc0 to dec

    def __init__(self):
        self.ip = "192.168.6.10"
        self.port = 1001
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dm_num = DM_NUM

    @staticmethod
    def _num_hex(num:int):
        hex_16 = hex(int(num))[2:].zfill(4)
        return hex_16[:2]+' '+hex_16[2:]

    @staticmethod
    def _voltage_hex(num:float, registry:int):
        _num = int((num+500)/1000*4096)
        _num = min(_num, 4095)
        _num = max(_num, 820)
        _num += REG_IDS[registry%4]
        hex_16 = DMUdp._num_hex(_num)
        return hex_16

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
        path = findlibs.find('Drv_UDPST')
        if path is None:
            raise Exception("Drv_UDPST.dll not found.")
        
        dll = cdll.LoadLibrary(path)
        dll.SetVoltages.restype = c_bool
        dll.SetVoltages.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double, ndim=1, shape=(self.dm_num)), c_int32, c_int32]

        dll.SetVoltagesNoEcho.restype = c_bool
        dll.SetVoltagesNoEcho.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double, ndim=1, shape=(self.dm_num)), c_int32, c_int32]

        dll.SetHV.restype = c_bool
        dll.SetHV.argtypes = [c_bool, c_bool]

        dll.GetVoltages.restype = c_bool
        dll.GetVoltages.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double, ndim=1, shape=(self.dm_num)), c_int32, c_int32]

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

    def get_voltages(self):
        voltages = np.zeros(self.dm_num, dtype=np.double)
        if self._dll.GetVoltages(voltages, c_int32(0), c_int32(self.dm_num)):
            return voltages
        else:
            raise Exception("device connection error.")




if __name__ == '__main__':
    def test():
        import os
        import tqdm
        
        v_dump_path = os.path.join(os.path.dirname(__file__), 'to_load_V.csv')
        # v = np.loadtxt(v_dump_path)
        with NLight(keep_when_exit=True) as dm:
            for i in tqdm.trange(100_000):
                v = np.zeros((dm.DM_Num,))
                v[1] = 30 * np.sin(2* np.pi * (i/10))
                dm.send_voltages(v, 0.2)
                
    def turn_off():
        with NLight(keep_when_exit=False) as dm:
            dm.reset_all()
            
    turn_off()