"""
BMS simulator that runs entirely inside the Streamlit process - no
separate terminal, script, or virtual serial port needed. This lets
the "Live BMS" flow be tested without any real hardware, all within
a single `streamlit run app.py` session.

When real hardware is available, SerialBMSDataSource is used in its
place - the interface is the same, so the UI/grading code doesn't
need to change.
"""

import random
import time
from typing import List

from data_sources.base import BatteryDataSource, BatteryReading, CellSnapshot
from data_sources.synthetic_cells import generate_demo_cell_snapshots

# Presets - same conditions as in sample_data
_PRESETS = {
    "Healthy battery": {"soh_target": 0.95, "ir": 0.012},
    "Medium wear battery": {"soh_target": 0.78, "ir": 0.018},
    "Degraded battery": {"soh_target": 0.55, "ir": 0.028},
}


class SimulatedBMSDataSource(BatteryDataSource):
    def __init__(
        self,
        preset: str = "Medium wear battery",
        cycle_count: int = 400,
        rated_capacity_ah: float = 100.0,
        num_readings: int = 15,
        num_cells: int = 16,
    ):
        self.preset = _PRESETS.get(preset, _PRESETS["Medium wear battery"])
        self.preset_name = preset if preset in _PRESETS else "Medium wear battery"
        self.cycle_count = cycle_count
        self.rated_capacity_ah = rated_capacity_ah
        self.num_readings = num_readings
        self.num_cells = num_cells
        # Simulated per-cell voltages, for the demo version of the
        # per-cell diagnostics feature - never real hardware data, see
        # synthetic_cells.py.
        self.cell_snapshots: List[CellSnapshot] = []

    def connect(self) -> None:
        pass  # nothing to connect to - everything is in-memory

    def read_all(self) -> List[BatteryReading]:
        self.cell_snapshots = generate_demo_cell_snapshots(
            condition=self.preset_name, num_cells=self.num_cells
        )

        readings = []
        current_capacity = self.rated_capacity_ah * self.preset["soh_target"]
        ir = self.preset["ir"]
        timestamp = time.time()

        for i in range(self.num_readings):
            readings.append(
                BatteryReading(
                    timestamp=timestamp + i * 5,
                    voltage=round(random.uniform(3.6, 4.1), 3),
                    current=round(random.uniform(-2.0, 2.0), 3),
                    temperature=round(random.uniform(28, 38), 1),
                    cycle_count=self.cycle_count,
                    capacity_ah=round(current_capacity + random.uniform(-0.15, 0.15), 3),
                    rated_capacity_ah=self.rated_capacity_ah,
                    internal_resistance=round(ir + random.uniform(-0.0015, 0.0015), 4),
                )
            )
        return readings
