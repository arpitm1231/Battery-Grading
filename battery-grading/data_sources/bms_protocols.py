"""
Binary wire protocols for common off-the-shelf BMS boards.

This module knows how to build the request bytes and parse the
response bytes for two widely-used BMS protocols, so
`SerialBMSDataSource` can talk to real hardware that doesn't speak
our simple JSON-lines format:

  - Daly UART/485 (protocol="daly")   - very common on Daly Smart BMS boards
  - JBD / Overkill Solar / "LLT" UART (protocol="jbd") - common on
    Overkill Solar, Xiaoxiang, and many rebranded 4S-24S LiFePO4 BMS
    boards (this is the same protocol referred to as "LltJbd" in
    other open-source BMS tools)

Each protocol here is documented from its public/reverse-engineered
spec (Daly's official "UART/485 Communications Protocol V1.2", and
the JBD "Smart BMS Protocol" doc used by Overkill Solar's tools).
Exact register layouts can vary slightly between firmware revisions,
so if your specific board doesn't match, `_parse_line` in
bms_source.py is the place to adjust the byte offsets.

NOT covered here (different enough to need dedicated work):
  - JK-BMS (Jikong) - has multiple, mutually-incompatible protocol
    versions (JK02/JK04, 24S/32S) depending on firmware, so it isn't
    safe to guess a single byte layout without testing against the
    actual unit.
  - CAN-bus BMS (e.g. real EV packs) - needs `python-can` + a
    manufacturer DBC file, which is a different transport entirely
    (not serial/UART).
If you have either of these, say so and the corresponding parser can
be added and tested against your unit's actual output.
"""

import struct
from typing import List, Optional


# ---------------------------------------------------------------------------
# Daly UART/485 protocol
# ---------------------------------------------------------------------------
# Frame format (both request and response) is 13 bytes:
#   [0]      start flag, always 0xA5
#   [1]      address (0x40 = "upper computer" sending; 0x01 = BMS replying)
#   [2]      data ID (which register)
#   [3]      data length, always 0x08
#   [4:12]   8 bytes of payload (all zero in a read request)
#   [12]     checksum = (sum of bytes[0:12]) & 0xFF

DALY_START = 0xA5
DALY_HOST_ADDR = 0x40
DALY_BMS_ADDR = 0x01

DALY_ID_VOLTAGE_CURRENT_SOC = 0x90
DALY_ID_MIN_MAX_TEMP = 0x92
DALY_ID_STATUS_CAPACITY = 0x93
DALY_ID_CELL_VOLTAGES = 0x95
DALY_ID_BALANCE_STATUS = 0x97


def _daly_checksum(payload: bytes) -> int:
    return sum(payload) & 0xFF


def build_daly_request(data_id: int) -> bytes:
    """Builds a 13-byte Daly read request for the given data ID."""
    body = bytes([DALY_START, DALY_HOST_ADDR, data_id, 0x08]) + bytes(8)
    return body + bytes([_daly_checksum(body)])


def parse_daly_frame(frame: bytes) -> Optional[dict]:
    """Parses a 13-byte Daly response frame. Returns None if the frame
    doesn't look valid (wrong length, bad start byte, bad checksum) -
    the caller should treat that as "no reading this cycle", not a
    fatal error, since a dropped/garbled frame is normal on a live
    serial line."""
    if len(frame) != 13 or frame[0] != DALY_START:
        return None
    if _daly_checksum(frame[:12]) != frame[12]:
        return None

    data_id = frame[2]
    data = frame[4:12]

    if data_id == DALY_ID_VOLTAGE_CURRENT_SOC:
        voltage_dv, _gather_dv, current_raw, soc_dpct = struct.unpack(
            ">HHHH", data
        )
        return {
            "data_id": data_id,
            "voltage": voltage_dv / 10.0,
            "current": (current_raw - 30000) / 10.0,
            "soc_percent": soc_dpct / 10.0,
        }

    if data_id == DALY_ID_STATUS_CAPACITY:
        state, charge_mos, discharge_mos, cycles = data[0:4]
        remaining_mah = struct.unpack(">I", data[4:8])[0]
        return {
            "data_id": data_id,
            "cycle_count": cycles,
            "capacity_ah": remaining_mah / 1000.0,
        }

    if data_id == DALY_ID_MIN_MAX_TEMP:
        max_temp_c = data[0] - 40
        min_temp_c = data[2] - 40
        return {
            "data_id": data_id,
            "temperature": (max_temp_c + min_temp_c) / 2.0,
        }

    return {"data_id": data_id}


def parse_daly_cell_voltage_frame(frame: bytes) -> Optional[tuple]:
    """
    Parses ONE frame of a Daly cell-voltage (0x95) response. The BMS
    doesn't answer with a single frame - it streams back one frame per
    3 cells (up to 16 frames for 48 cells), each still 13 bytes:
      data[0]   frame number, starting at 0 (0xFF = no more frames)
      data[1:7] 3 cell voltages, 2 bytes each, in millivolts
      data[7]   reserved

    Returns (frame_number, [v1, v2, v3]) in volts, or None if the
    frame is invalid or is the "no more data" terminator.
    """
    if len(frame) != 13 or frame[0] != DALY_START:
        return None
    if _daly_checksum(frame[:12]) != frame[12]:
        return None
    if frame[2] != DALY_ID_CELL_VOLTAGES:
        return None

    data = frame[4:12]
    frame_number = data[0]
    if frame_number == 0xFF:
        return None  # terminator - no more cells

    v1, v2, v3 = struct.unpack(">HHH", data[1:7])
    return (frame_number, [v1 / 1000.0, v2 / 1000.0, v3 / 1000.0])


def parse_daly_balance_status(frame: bytes, num_cells: int) -> Optional[List[bool]]:
    """
    Parses a Daly balance-status (0x97) response: an 8-byte (64-bit)
    bitmap, bit 0 = cell 1's balance state (1 = actively balancing).
    Returns a list of `num_cells` booleans, or None if the frame is
    invalid.
    """
    if len(frame) != 13 or frame[0] != DALY_START:
        return None
    if _daly_checksum(frame[:12]) != frame[12]:
        return None
    if frame[2] != DALY_ID_BALANCE_STATUS:
        return None

    data = frame[4:12]
    bitmap = int.from_bytes(data, byteorder="little")
    return [bool(bitmap & (1 << i)) for i in range(num_cells)]


# ---------------------------------------------------------------------------
# JBD / Overkill Solar / "LLT" UART protocol
# ---------------------------------------------------------------------------
# Request is a fixed 7-byte command (these exact bytes are the
# well-known, widely-used "read basic info" and "read cell voltages"
# commands for this protocol):
JBD_REQUEST_BASIC_INFO = bytes.fromhex("DDA5030000FFFD77")
JBD_REQUEST_CELL_VOLTAGES = bytes.fromhex("DDA5040000FFFC77")

JBD_START = 0xDD
JBD_STOP = 0x77


def parse_jbd_basic_info(frame: bytes) -> Optional[dict]:
    """
    Parses a JBD "basic info" (command 0x03) response frame:
      [0]        start byte, 0xDD
      [1]        echoed command, 0x03
      [2]        status (0 = OK)
      [3]        data length N
      [4:4+N]    data payload
      [4+N:6+N]  checksum (2 bytes, not strictly validated here since
                 exact firmware behavior varies slightly - a garbled
                 frame will fail to parse the fields below instead)
      [6+N]      stop byte, 0x77

    Data payload layout (byte offsets within the payload):
      0-1   total voltage        (u16, unit 0.01V)
      2-3   current              (s16, unit 0.01A, +charge/-discharge)
      4-5   residual capacity    (u16, unit 0.01Ah)
      6-7   nominal capacity     (u16, unit 0.01Ah)  <- used as rated capacity
      8-9   cycle count          (u16)
      ...   (production date, balance/protection status, version, etc.)
      21    number of battery cells/strings
      22    number of temperature sensors (NTCs)
      23+   one u16 per sensor, unit 0.1K, offset by 2731 (i.e. subtract
            2731 and divide by 10 to get Celsius)
    """
    if len(frame) < 7 or frame[0] != JBD_START or frame[-1] != JBD_STOP:
        return None
    if frame[1] != 0x03 or frame[2] != 0x00:
        return None  # wrong command echoed back, or BMS reported an error

    length = frame[3]
    payload = frame[4:4 + length]
    if len(payload) < 10:
        return None  # too short to contain the fields we need

    voltage_raw, current_raw, residual_raw, nominal_raw, cycles = struct.unpack(
        ">HhHHH", payload[0:10]
    )

    temperature = None
    if len(payload) > 22:
        num_ntc = payload[22]
        if num_ntc > 0 and len(payload) >= 23 + 2 * num_ntc:
            temps = []
            for i in range(num_ntc):
                raw_t = struct.unpack(
                    ">H", payload[23 + 2 * i: 25 + 2 * i]
                )[0]
                temps.append((raw_t - 2731) / 10.0)
            temperature = sum(temps) / len(temps)

    return {
        "voltage": voltage_raw / 100.0,
        "current": current_raw / 100.0,
        "capacity_ah": residual_raw / 100.0,
        "rated_capacity_ah": nominal_raw / 100.0,
        "cycle_count": cycles,
        "temperature": temperature,
    }


def parse_jbd_cell_voltages(frame: bytes) -> Optional[List[float]]:
    """
    Parses a JBD "cell voltages" (command 0x04) response frame:
      [0]        start byte, 0xDD
      [1]        echoed command, 0x04
      [2]        status (0 = OK)
      [3]        data length N (= 2 * number of cells)
      [4:4+N]    one u16 (millivolts) per cell, high byte first
      ...        checksum (2 bytes) + stop byte (0x77)

    Returns a list of cell voltages in volts, or None if the frame is
    invalid.
    """
    if len(frame) < 7 or frame[0] != JBD_START or frame[-1] != JBD_STOP:
        return None
    if frame[1] != 0x04 or frame[2] != 0x00:
        return None

    length = frame[3]
    payload = frame[4:4 + length]
    if len(payload) != length or length % 2 != 0 or length == 0:
        return None

    num_cells = length // 2
    raw_values = struct.unpack(f">{num_cells}H", payload)
    return [v / 1000.0 for v in raw_values]
