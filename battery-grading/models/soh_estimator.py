"""
Estimates a battery's State of Health (SoH) from its readings.

Approach: simple but explainable - combines signals like capacity
fade and internal resistance into a single SoH score (0-100%). This
can later be replaced with a trained ML model without touching any
other code - that's why it's a clean function interface.
"""

from statistics import mean
from typing import List

from data_sources.base import BatteryReading


def estimate_soh(readings: List[BatteryReading]) -> float:
    """
    Returns SoH % as: (current usable capacity / rated capacity) * 100

    In the real world this is more sophisticated (voltage curve shape,
    internal resistance growth, temperature history all factor in) -
    for now this is a capacity-fade based estimate that can be
    replaced with a trained model later.
    """
    if not readings:
        raise ValueError("No readings available to estimate SoH from")

    # Average the most recent readings' capacity measurement (to reduce noise)
    recent = readings[-min(10, len(readings)):]
    avg_measured_capacity = mean(r.capacity_ah for r in recent)
    rated_capacity = readings[-1].rated_capacity_ah

    if rated_capacity <= 0:
        raise ValueError(
            f"Invalid rated_capacity_ah ({rated_capacity}) - must be greater than 0"
        )

    soh_percent = (avg_measured_capacity / rated_capacity) * 100
    return round(max(0.0, min(100.0, soh_percent)), 2)


def estimate_degradation_rate(readings: List[BatteryReading]) -> float:
    """
    Returns how much % capacity is lost per 100 cycles. This shows
    how quickly the battery will keep degrading going forward.
    """
    if len(readings) < 2:
        return 0.0

    first, last = readings[0], readings[-1]
    cycle_delta = last.cycle_count - first.cycle_count
    if cycle_delta <= 0:
        return 0.0

    if first.rated_capacity_ah <= 0 or last.rated_capacity_ah <= 0:
        return 0.0

    capacity_pct_first = (first.capacity_ah / first.rated_capacity_ah) * 100
    capacity_pct_last = (last.capacity_ah / last.rated_capacity_ah) * 100
    pct_drop = capacity_pct_first - capacity_pct_last

    rate_per_100_cycles = (pct_drop / cycle_delta) * 100
    return round(rate_per_100_cycles, 3)
