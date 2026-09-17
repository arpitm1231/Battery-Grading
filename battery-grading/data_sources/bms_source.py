"""
Data source that reads live data from a real battery BMS over a
serial (UART) connection.

Three wire protocols are supported (pick one with the `protocol`
argument):

  - "json"  (default) - one JSON object per line, e.g. from a DIY
    Arduino/ESP32 setup. See PROTOCOL below for the exact format.
  - "daly"  - Daly Smart BMS UART/485 protocol (very common on Daly
    BMS boards).
  - "jbd"   - JBD / Overkill Solar / "LLT" UART protocol (common on
    Overkill Solar, Xiaoxiang, and many rebranded LiFePO4 BMS boards).

The byte-level details of "daly" and "jbd" live in bms_protocols.py.
If your BMS uses something else entirely (e.g. JK-BMS, or a CAN-bus
pack), see the note at the top of bms_protocols.py - those need
protocol-specific work rather than a generic parser.

JSON PROTOCOL: the BMS (or a microcontroller connected to it, like an
Arduino/ESP32) sends one JSON object per line over the serial port,
in this format:

    {"voltage": 3.72, "current": 1.15, "temperature": 28.4,
     "cycle_count": 150, "capacity_ah": 95.2, "rated_capacity_ah": 100,
     "internal_resistance": 0.015}

`internal_resistance` is optional (omit the field or send null if you
can't provide it) - the other fields are required.

NOTE: this module requires the `pyserial` package (`pip install
pyserial`). The import is done lazily inside each function so the
rest of the app keeps working - without pyserial installed - for
anyone who isn't using real hardware.
"""

import json
import time
from typing import List, Optional

from data_sources.base import BatteryDataSource, BatteryReading, CellSnapshot
from data_sources.bms_protocols import (
    DALY_ID_BALANCE_STATUS,
    DALY_ID_CELL_VOLTAGES,
    DALY_ID_MIN_MAX_TEMP,
    DALY_ID_STATUS_CAPACITY,
    DALY_ID_VOLTAGE_CURRENT_SOC,
    JBD_REQUEST_BASIC_INFO,
    JBD_REQUEST_CELL_VOLTAGES,
    build_daly_request,
    parse_daly_balance_status,
    parse_daly_cell_voltage_frame,
    parse_daly_frame,
    parse_jbd_basic_info,
    parse_jbd_cell_voltages,
)

SUPPORTED_PROTOCOLS = ("json", "daly", "jbd")


class SerialLibraryMissingError(ImportError):
    """Raised when pyserial isn't installed but real-hardware mode was
    requested. Kept as a distinct type so callers (e.g. the UI) can
    catch it specifically and show a helpful message instead of a
    raw traceback."""
    pass


def list_serial_ports() -> List[str]:
    """Returns a list of all serial ports connected to the computer -
    useful for building a dropdown in the UI (no port needs to be
    hardcoded)."""
    try:
        import serial.tools.list_ports
    except ImportError as e:
        raise SerialLibraryMissingError(
            "The 'pyserial' package is not installed. Run "
            "'pip install pyserial' and restart the app to use real "
            "hardware mode."
        ) from e

    return [p.device for p in serial.tools.list_ports.comports()]


class SerialBMSDataSource(BatteryDataSource):
    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        read_duration_seconds: float = 10.0,
        min_samples: int = 1,
        timeout: float = 2.0,
        protocol: str = "json",
        rated_capacity_ah: Optional[float] = None,
    ):
        if protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(
                f"Unknown protocol '{protocol}'. Supported protocols: "
                f"{', '.join(SUPPORTED_PROTOCOLS)}"
            )
        self.port = port
        self.baudrate = baudrate
        self.read_duration_seconds = read_duration_seconds
        self.min_samples = min_samples
        self.timeout = timeout
        self.protocol = protocol
        # Only used by protocols that don't report a nameplate/rated
        # capacity themselves (e.g. Daly) - for those, the caller
        # (the UI) supplies this since it's a fixed, known constant
        # rather than something read off the BMS each cycle.
        self.rated_capacity_ah = rated_capacity_ah
        self._connection = None
        # Populated during read_all() for protocols that expose
        # per-cell data (Daly, JBD) - one entry per successful poll of
        # all cells during the session. Used for cell-level
        # diagnostics (e.g. spotting a single weak/outlier cell).
        # Stays empty for the JSON protocol, which has no per-cell data.
        self.cell_snapshots: List[CellSnapshot] = []

    def connect(self) -> None:
        try:
            import serial
        except ImportError as e:
            raise SerialLibraryMissingError(
                "The 'pyserial' package is not installed. Run "
                "'pip install pyserial' and restart the app to use real "
                "hardware mode."
            ) from e

        try:
            self._connection = serial.Serial(
                self.port, self.baudrate, timeout=self.timeout
            )
            # Right after opening, a freshly-opened serial port resets
            # the microcontroller (if it's an Arduino) - wait a moment
            # for it to boot back up.
            time.sleep(2.0)
        except serial.SerialException as e:
            raise ConnectionError(
                f"Could not connect to '{self.port}' at {self.baudrate} baud. "
                f"Check that: the correct port is selected, the BMS/adapter "
                f"is physically connected, and no other software (like the "
                f"Arduino IDE Serial Monitor) already has that port open. "
                f"Original error: {e}"
            ) from e

    def read_all(self) -> List[BatteryReading]:
        if self._connection is None:
            self.connect()

        self.cell_snapshots = []

        if self.protocol == "json":
            readings = self._read_all_json()
        elif self.protocol == "daly":
            readings = self._read_all_daly()
        elif self.protocol == "jbd":
            readings = self._read_all_jbd()
        else:
            raise ValueError(f"Unknown protocol '{self.protocol}'")

        if len(readings) < self.min_samples:
            raise ValueError(
                f"Only {len(readings)} valid reading(s) received in "
                f"{self.read_duration_seconds}s (need at least "
                f"{self.min_samples}). Check the wiring, baud rate, and "
                f"that '{self.protocol}' is the correct protocol for this "
                f"BMS."
            )

        return readings

    # -- JSON-lines protocol -------------------------------------------------

    def _read_all_json(self) -> List[BatteryReading]:
        readings: List[BatteryReading] = []
        start_time = time.time()

        while (time.time() - start_time) < self.read_duration_seconds:
            raw_line = self._connection.readline()
            if not raw_line:
                continue  # timed out, no data arrived this cycle

            reading = self._parse_json_line(raw_line)
            if reading is not None:
                readings.append(reading)

        return readings

    def _parse_json_line(self, raw_line: bytes) -> Optional[BatteryReading]:
        """Converts one line into a BatteryReading. Silently skips
        corrupt/partial lines (which are common in serial
        communication) instead of crashing the whole read - this is
        necessary to stay robust with real hardware."""
        try:
            text = raw_line.decode("utf-8", errors="ignore").strip()
            if not text:
                return None
            data = json.loads(text)
            return BatteryReading(
                timestamp=data.get("timestamp", time.time()),
                voltage=float(data["voltage"]),
                current=float(data["current"]),
                temperature=float(data["temperature"]),
                cycle_count=int(data["cycle_count"]),
                capacity_ah=float(data["capacity_ah"]),
                rated_capacity_ah=float(data["rated_capacity_ah"]),
                internal_resistance=(
                    float(data["internal_resistance"])
                    if data.get("internal_resistance") is not None
                    else None
                ),
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            # Corrupt line - normal in serial communication (baud rate
            # mismatch, cable noise, partial line boundary). Skip it
            # silently rather than failing the whole connection.
            return None

    # -- Daly UART/485 protocol -----------------------------------------------

    def _read_all_daly(self) -> List[BatteryReading]:
        readings: List[BatteryReading] = []
        start_time = time.time()

        while (time.time() - start_time) < self.read_duration_seconds:
            vcs = self._daly_query(DALY_ID_VOLTAGE_CURRENT_SOC)
            status = self._daly_query(DALY_ID_STATUS_CAPACITY)
            temp = self._daly_query(DALY_ID_MIN_MAX_TEMP)

            if vcs is not None and status is not None:
                rated = self.rated_capacity_ah or status["capacity_ah"]
                readings.append(
                    BatteryReading(
                        timestamp=time.time(),
                        voltage=vcs["voltage"],
                        current=vcs["current"],
                        temperature=temp["temperature"] if temp else 25.0,
                        cycle_count=status["cycle_count"],
                        capacity_ah=status["capacity_ah"],
                        rated_capacity_ah=rated,
                        internal_resistance=None,  # not exposed by this protocol
                    )
                )

            cells = self._daly_read_cell_voltages()
            if cells:
                balancing = self._daly_query_balance_status(len(cells))
                self.cell_snapshots.append(
                    CellSnapshot(
                        timestamp=time.time(), voltages=cells, balancing=balancing
                    )
                )

            time.sleep(0.3)

        return readings

    def _daly_query(self, data_id: int) -> Optional[dict]:
        try:
            if hasattr(self._connection, "reset_input_buffer"):
                self._connection.reset_input_buffer()
            self._connection.write(build_daly_request(data_id))
            response = self._connection.read(13)
            return parse_daly_frame(response)
        except Exception:
            # A dropped byte or timeout on one query shouldn't kill the
            # whole reading session - just skip this cycle's value.
            return None

    def _daly_read_cell_voltages(self) -> List[float]:
        """Sends one 0x95 request; the BMS streams back one 13-byte
        frame per 3 cells until a terminator (frame number 0xFF) or
        the stream simply stops. Collects and orders every cell
        voltage received."""
        cells_by_frame: dict = {}
        try:
            if hasattr(self._connection, "reset_input_buffer"):
                self._connection.reset_input_buffer()
            self._connection.write(build_daly_request(DALY_ID_CELL_VOLTAGES))

            for _ in range(16):  # at most 16 frames (48 cells / 3 per frame)
                frame = self._connection.read(13)
                if len(frame) < 13:
                    break  # no more frames arriving
                parsed = parse_daly_cell_voltage_frame(frame)
                if parsed is None:
                    break  # terminator frame or a corrupt frame - stop here
                frame_number, voltages = parsed
                cells_by_frame[frame_number] = voltages
        except Exception:
            pass  # return whatever was collected before the failure

        cells: List[float] = []
        for frame_number in sorted(cells_by_frame):
            cells.extend(cells_by_frame[frame_number])
        return cells

    def _daly_query_balance_status(self, num_cells: int) -> Optional[List[bool]]:
        try:
            if hasattr(self._connection, "reset_input_buffer"):
                self._connection.reset_input_buffer()
            self._connection.write(build_daly_request(DALY_ID_BALANCE_STATUS))
            response = self._connection.read(13)
            return parse_daly_balance_status(response, num_cells)
        except Exception:
            return None

    # -- JBD / Overkill Solar / "LLT" protocol ---------------------------------

    def _read_all_jbd(self) -> List[BatteryReading]:
        readings: List[BatteryReading] = []
        start_time = time.time()

        while (time.time() - start_time) < self.read_duration_seconds:
            parsed = self._jbd_query_basic_info()
            if parsed is not None:
                rated = self.rated_capacity_ah or parsed["rated_capacity_ah"]
                readings.append(
                    BatteryReading(
                        timestamp=time.time(),
                        voltage=parsed["voltage"],
                        current=parsed["current"],
                        temperature=(
                            parsed["temperature"]
                            if parsed["temperature"] is not None
                            else 25.0
                        ),
                        cycle_count=parsed["cycle_count"],
                        capacity_ah=parsed["capacity_ah"],
                        rated_capacity_ah=rated,
                        internal_resistance=None,  # not exposed by this protocol
                    )
                )

            cells = self._jbd_query_cell_voltages()
            if cells:
                self.cell_snapshots.append(
                    CellSnapshot(timestamp=time.time(), voltages=cells, balancing=None)
                )

            time.sleep(0.3)

        return readings

    def _jbd_query_basic_info(self) -> Optional[dict]:
        try:
            if hasattr(self._connection, "reset_input_buffer"):
                self._connection.reset_input_buffer()
            self._connection.write(JBD_REQUEST_BASIC_INFO)

            header = self._connection.read(4)
            if len(header) < 4:
                return None
            length = header[3]
            rest = self._connection.read(length + 3)
            if len(rest) < length + 3:
                return None

            return parse_jbd_basic_info(header + rest)
        except Exception:
            return None

    def _jbd_query_cell_voltages(self) -> Optional[List[float]]:
        try:
            if hasattr(self._connection, "reset_input_buffer"):
                self._connection.reset_input_buffer()
            self._connection.write(JBD_REQUEST_CELL_VOLTAGES)

            header = self._connection.read(4)
            if len(header) < 4:
                return None
            length = header[3]
            rest = self._connection.read(length + 3)
            if len(rest) < length + 3:
                return None

            return parse_jbd_cell_voltages(header + rest)
        except Exception:
            return None

    def close(self) -> None:
        if self._connection is not None and self._connection.is_open:
            self._connection.close()
            self._connection = None
