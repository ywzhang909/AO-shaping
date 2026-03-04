from ao_shaping.drivers.tm import TM

import serial
import time
import pytest


def test_serial():
    """Test serial port communication - requires hardware"""
    pytest.skip("Requires serial port hardware")


def test_pack_position_xy():
    """Test the pack_position_xy method"""
    # Test normal values
    result = TM.pack_position_xy(8.0, 16.0)
    expected = bytes([0x00, 0xA0, 0x01, 0x40])  # Big-endian format
    assert result == expected, f"Expected {expected.hex()}, got {result.hex()}"

    # Test negative values
    result = TM.pack_position_xy(-8.0, -16.0)
    # -8.0 / 0.05 = -160 = 0xFF60 in 16-bit two's complement
    # -16.0 / 0.05 = -320 = 0xFEC0 in 16-bit two's complement
    expected = bytes([0xFF, 0x60, 0xFE, 0xC0])  # Big-endian format
    assert result == expected, f"Expected {expected.hex()}, got {result.hex()}"

    print("All pack_position_xy tests passed!")


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
    checksum = ~(sum(frame[2:12])) & 0xFF
    assert frame[12] == checksum, (
        f"Checksum mismatch: expected {checksum:02X}, got {frame[12]:02X}"
    )

    print("All _build_frame tests passed!")


def test_range_limiting():
    """Test range limiting functionality"""
    # Test values within range
    frame = TM._build_frame(100.0, -100.0)
    print(frame)
    # Should not be modified
    assert frame[3:7] == TM.pack_position_xy(100.0, -100.0)

    print("All range limiting tests passed!")


def test_port_list():
    print(TM.list_port())


def test_tm():
    """Test TM (tilt mirror) - requires hardware"""
    pytest.skip("Requires TM hardware")
