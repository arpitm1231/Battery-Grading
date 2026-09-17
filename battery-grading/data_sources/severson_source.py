"""
Converts research-grade battery cycling datasets (like the MIT/Stanford/
Toyota fast-charging dataset) into our BatteryReading format.

This format is different from normal BMS logs - it's a per-cycle
summary (IR, QC, QD, Tavg, chargetime) rather than a raw
voltage/current time-series. So the mapping is a bit different:

  capacity_ah         <- QD (discharge capacity, best proxy for usable capacity)
  rated_capacity_ah    <- QD of the first valid cycle (nominal capacity)
  internal_resistance  <- IR (available directly - the most valuable
                           signal in this dataset, which normal BMS
                           logs don't have)
  temperature          <- Tavg
  voltage/current       <- not present in this dataset, default to 0.0

Cycle 1 (a warm-up cycle, IR=0/QD=0) is skipped since it's invalid.
"""

import csv
from collections import defaultdict
from typing import Dict, List

from data_sources.base import BatteryDataSource, BatteryReading

# Columns that confirm a file is in this format
EXPECTED_COLUMNS = {"IR", "QC", "QD", "Tavg", "cycle", "battery_id"}


def is_severson_format(file_path: str) -> bool:
    """Checks the file's header to tell whether it's in this
    (raw lab dataset) format or our simple BMS format."""
    with open(file_path, "r") as f:
        header = set(f.readline().strip().split(","))
    return EXPECTED_COLUMNS.issubset(header)


class SeversonCSVDataSource(BatteryDataSource):
    def __init__(self, file_path: str, battery_id: str):
        self.file_path = file_path
        self.battery_id = battery_id
        self._connected = False

    def connect(self) -> None:
        with open(self.file_path, "r") as f:
            pass
        self._connected = True

    def list_battery_ids(self) -> List[str]:
        """Returns the IDs of every battery present in the file -
        useful for building a dropdown in the UI."""
        ids = []
        seen = set()
        with open(self.file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bid = row["battery_id"]
                if bid not in seen:
                    seen.add(bid)
                    ids.append(bid)
        return ids

    def read_all(self) -> List[BatteryReading]:
        if not self._connected:
            self.connect()

        readings = []
        rated_capacity = None

        with open(self.file_path, "r") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r["battery_id"] == self.battery_id]

        # Sort by cycle number so ordering is guaranteed
        rows.sort(key=lambda r: float(r["cycle"]))

        for row in rows:
            cycle = float(row["cycle"])
            if cycle <= 1:
                continue  # skip the warm-up cycle, it's invalid

            qd = float(row["QD"])
            if qd <= 0:
                continue

            if rated_capacity is None:
                rated_capacity = qd  # first valid cycle = nominal capacity

            ir = float(row["IR"]) if row["IR"] not in ("", None) else None

            readings.append(
                BatteryReading(
                    timestamp=cycle * float(row.get("chargetime", 0) or 0),
                    voltage=0.0,   # per-cycle voltage isn't in this dataset
                    current=0.0,
                    temperature=float(row["Tavg"]),
                    cycle_count=int(cycle),
                    capacity_ah=qd,
                    rated_capacity_ah=rated_capacity,
                    internal_resistance=ir,
                )
            )
        return readings
