from ao_shaping.drivers.tm import TM

import serial
import time
import pytest

@pytest.mark.experiment
def test_serial():
    ser = serial.Serial(timeout=0.5)
    ser.port = 'COM6'
    ser.baudrate = 9600
    ser.bytesize = serial.EIGHTBITS
    ser.parity = serial.PARITY_NONE
    ser.stopbits = serial.STOPBITS_ONE

    ser.open()

    send_msg = bytes([0x01])
    ser.write(send_msg)
    time.sleep(0.1)
    response = ser.read(1)
    print(response)
    assert response == send_msg, f"Expected {send_msg.hex()}, got {response.hex()}"

    ser.close()


@pytest.mark.experiment
def test_pack_position_xy():
    """Test the pack_position_xy method"""
    # Test normal values
    result = TM.pack_position_xy(8.0, 16.0)
    expected = bytes([0x00, 0xA0, 0x01, 0x40])  # Big-endian format
    assert result == expected, f"Expected {expected.hex()}, got {result.hex()}"

    # Test negative values
    result = TM.pack_position_xy(-8.0, -16.0)
    # -8.0 / 0.05 = -160 = 0xFF60 in 16-bit two's complement
    # -16.0 / 0.05 = -320 = 0xFE80 in 16-bit two's complement
    expected = bytes([0xFF, 0x60, 0xFE, 0x80])  # Big-endian format
    assert result == expected, f"Expected {expected.hex()}, got {result.hex()}"

    print("All pack_position_xy tests passed!")

@pytest.mark.experiment
def test_build_frame():
    """Test the _build_frame method"""
    # Test normal values
    frame = TM._build_frame(8.0, 16.0)
    expected_header = bytes([0x7E, 0xE7, 0x01])
    expected_position = bytes([0x00, 0xA0, 0x01, 0x40])  # X=0x00A0, Y=0x0140
    expected_zeros = bytes([0x00, 0x00, 0x00, 0x00, 0x00])

    assert frame[0:3] == expected_header, f"Header mismatch"
    assert frame[3:7] == expected_position, f"Position data mismatch"
    assert frame[7:12] == expected_zeros, f"Zero padding mismatch"

    # Verify checksum
    checksum = (~(sum(frame[2:12])) & 0xFF)
    assert frame[12] == checksum, f"Checksum mismatch: expected {checksum:02X}, got {frame[12]:02X}"

    print("All _build_frame tests passed!")

@pytest.mark.experiment
def test_range_limiting():
    """Test range limiting functionality"""
    # Test values within range
    frame = TM._build_frame(100.0, -100.0)
    print(frame)
    # Should not be modified
    assert frame[3:7] == TM.pack_position_xy(100.0, -100.0)

    print("All range limiting tests passed!")


@pytest.mark.experiment
def test_port_list():
    print(TM.list_port())

@pytest.mark.experiment
def test_tm():
    position = (0, -420)
    # position = (412, 389)
    with TM("COM3") as tm:
        frame = TM._build_frame(*position)
        tm.send(*position)
        ret_frame = tm.wait_rx()
        # bin to hex
        ret_pos = tm.bin_frame_to_pos(ret_frame[2:6])
        print(ret_pos, frame, ret_frame)
        # assert ret_pos == (0.0, 51.150000000000006)
