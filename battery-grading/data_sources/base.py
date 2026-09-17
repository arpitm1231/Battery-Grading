"""
Base interface for all battery data sources.

Core idea: whether data comes from a CSV file or a real BMS (Serial/CAN),
the grading logic shouldn't care. Every data source follows this
interface, so the code above (grading, models) never has to change
when the data source changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BatteryReading:
    """
    A single reading from a battery - whether it came from a CSV row
    or a real BMS, it should arrive in this same format.

    internal_resistance is optional: a basic BMS (only voltage/current/
    temp) will leave this as None and the simple capacity-ratio logic
    will run. A data source that also provides IR (like lab-grade
    cyclers or advanced BMS units) unlocks the ML-based robust
    prediction.
    """
    timestamp: float          # seconds
    voltage: float            # volts
    current: float            # amps (+ve = charging, -ve = discharging)
    temperature: float        # celsius
    cycle_count: int          # total charge-discharge cycles so far
    capacity_ah: float        # measured capacity in this cycle (amp-hours)
    rated_capacity_ah: float  # original/nameplate capacity (amp-hours)
    internal_resistance: Optional[float] = None  # ohms, if available


@dataclass
class CellSnapshot:
    """
    One "poll" of every individual cell's voltage, taken during a real
    BMS read session. Only protocols that expose per-cell data (Daly,
    JBD) produce these - the simple JSON-lines protocol, CSV files,
    and the simulator only have pack-level data, so this stays unused
    for them.

    `balancing` (if available) marks which cells the BMS is actively
    bleeding down to match the rest of the pack - a cell that balances
    unusually often is one signal (not the only one) that it's out of
    step with the others.
    """
    timestamp: float
    voltages: List[float]                       # one entry per cell, volts
    balancing: Optional[List[bool]] = None       # one entry per cell, if available


class BatteryDataSource(ABC):
    """
    Every data source (CSV, real BMS, simulator...) inherits this
    class and implements these 2 methods.
    """

    @abstractmethod
    def connect(self) -> None:
        """Set up the connection to the data source (open a file,
        open a serial port, etc.)"""
        raise NotImplementedError

    @abstractmethod
    def read_all(self) -> List[BatteryReading]:
        """Return all available readings as a list, in
        BatteryReading format."""
        raise NotImplementedError

    def close(self) -> None:
        """Optional cleanup - a subclass can override this if it
        needs to close a connection (like a serial port)."""
        pass
