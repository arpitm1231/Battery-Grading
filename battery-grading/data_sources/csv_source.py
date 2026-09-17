"""
Data source that reads battery data from a CSV file.
For development/testing - dataset files (like NASA/Oxford) or your
own generated sample data load through this.

Expected CSV columns:
timestamp, voltage, current, temperature, cycle_count, capacity_ah, rated_capacity_ah
"""

import csv
from typing import List

from data_sources.base import BatteryDataSource, BatteryReading

REQUIRED_COLUMNS = [
    "timestamp",
    "voltage",
    "current",
    "temperature",
    "cycle_count",
    "capacity_ah",
    "rated_capacity_ah",
]


class CSVDataSource(BatteryDataSource):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._connected = False

    def connect(self) -> None:
        # Just confirm the file exists and is readable
        with open(self.file_path, "r") as f:
            pass
        self._connected = True

    def read_all(self) -> List[BatteryReading]:
        if not self._connected:
            self.connect()

        readings = []
        with open(self.file_path, "r") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV file appears to be empty (no header row).")

            missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
            if missing:
                raise KeyError(
                    f"Missing required column(s): {', '.join(missing)}"
                )

            for i, row in enumerate(reader, start=2):
                try:
                    readings.append(
                        BatteryReading(
                            timestamp=float(row["timestamp"]),
                            voltage=float(row["voltage"]),
                            current=float(row["current"]),
                            temperature=float(row["temperature"]),
                            cycle_count=int(float(row["cycle_count"])),
                            capacity_ah=float(row["capacity_ah"]),
                            rated_capacity_ah=float(row["rated_capacity_ah"]),
                        )
                    )
                except (ValueError, TypeError) as e:
                    # Fail with a clear row number rather than a generic
                    # error, so a bad value is easy to find and fix.
                    raise ValueError(
                        f"Row {i} has an invalid/non-numeric value: {e}"
                    ) from e

        return readings
