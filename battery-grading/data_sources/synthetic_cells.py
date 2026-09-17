"""
Generates synthetic per-cell voltage data for demo purposes only -
used by the "Sample data" and "Simulated BMS" modes, which don't have
a real BMS to poll individual cells from.

IMPORTANT: this is never used for a user's uploaded CSV. Fabricating
cell-level detail for someone's real battery data would be
misleading - it's only meant to demonstrate what the per-cell
diagnostics feature looks like, and the UI clearly labels it as
simulated/demo data so it's never mistaken for a real reading.
"""

import random
from typing import List

from data_sources.base import CellSnapshot

# Presets - each names one cell to deliberately make "weak" for the
# demo (0 = no weak cell, i.e. a fully healthy pack)
_PRESETS = {
    "Healthy battery": {"weak_cell": None, "weak_offset_mv": 0, "noise_mv": 4},
    "Medium wear battery": {"weak_cell": 7, "weak_offset_mv": -45, "noise_mv": 5},
    "Degraded battery": {"weak_cell": 11, "weak_offset_mv": -90, "noise_mv": 7},
}


def generate_demo_cell_snapshots(
    condition: str,
    num_cells: int = 16,
    num_snapshots: int = 12,
    nominal_voltage: float = 3.30,
    seed: int = None,
) -> List[CellSnapshot]:
    """
    Builds a series of synthetic cell-voltage snapshots for one of the
    3 demo conditions. If `condition` isn't recognized, defaults to a
    healthy pack (no weak cell).
    """
    preset = _PRESETS.get(condition, _PRESETS["Healthy battery"])
    rng = random.Random(seed)

    snapshots = []
    for i in range(num_snapshots):
        voltages = [
            nominal_voltage + rng.uniform(-preset["noise_mv"], preset["noise_mv"]) / 1000.0
            for _ in range(num_cells)
        ]
        if preset["weak_cell"] is not None and preset["weak_cell"] <= num_cells:
            idx = preset["weak_cell"] - 1  # 1-based -> 0-based
            voltages[idx] += preset["weak_offset_mv"] / 1000.0
        snapshots.append(
            CellSnapshot(timestamp=float(i), voltages=voltages, balancing=None)
        )

    return snapshots
