"""
Per-cell diagnostics: finds which individual cell in a pack is
behaving differently from the rest - the one you'd actually want to
replace, instead of scrapping the whole pack.

Two layers, on purpose:

  1. SESSION-LEVEL (this connect-and-read session only): looks at all
     the cell-voltage snapshots collected during one real-hardware
     read and flags whichever cell is a statistical outlier from the
     others.

  2. CROSS-SESSION HISTORY (the real "does this keep happening"
     signal): a single 10-30 second session only shows short-term
     spread - it can't tell you if a cell is truly degrading over
     time. So every session's summary is appended to a small local
     JSON log file (keyed by which serial port/BMS was used), and
     future sessions read that history back to answer "has this same
     cell been flagged before, and how often?"

Only real BMS protocols that expose per-cell voltages (Daly, JBD) can
use this - CSV/sample/simulated data stays pack-level for now.
"""

import json
import os
import re
import statistics
from dataclasses import dataclass
from typing import List, Optional

from data_sources.base import CellSnapshot

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cell_health_log")

# A cell's average deviation from the pack has to clear BOTH of these
# to be flagged - this is what makes the threshold "robust" rather
# than either too twitchy or too blunt:
#   - MIN_ABSOLUTE_MV: a floor so that a genuinely tight/healthy pack
#     (where every cell is within a few mV of the others) doesn't get
#     a false alarm from ordinary measurement noise.
#   - a multiple of the pack's own spread (MAD, see below): so that a
#     pack that's naturally a bit more spread out doesn't cry wolf on
#     every cell - only a cell that's an outlier *relative to its own
#     pack* gets flagged.
MIN_ABSOLUTE_MV = 30.0
MAD_MULTIPLIER = 3.0


@dataclass
class CellStat:
    index: int                    # 1-based cell number, for display
    mean_voltage: float            # volts, averaged over the session
    deviation_mv: float             # (this cell's mean) - (pack mean), in mV
    balance_fraction: float         # 0-1, how often this cell was balancing
    is_outlier: bool


@dataclass
class SessionCellReport:
    num_cells: int
    cell_stats: List[CellStat]
    flagged_cell: Optional[CellStat]  # the single strongest outlier, if any
    num_snapshots: int


def analyze_session(snapshots: List[CellSnapshot]) -> Optional[SessionCellReport]:
    """
    Looks at every cell-voltage snapshot collected during one session
    and flags the cell (if any) that's a statistical outlier from the
    rest of the pack, averaged across the whole session (not just one
    reading - a single noisy sample shouldn't flag a healthy cell).
    """
    if not snapshots:
        return None

    num_cells = min(len(s.voltages) for s in snapshots)
    if num_cells < 2:
        return None  # can't compare a cell against "the rest" with <2 cells

    # Average deviation-from-pack-mean for each cell, across every snapshot
    per_cell_deviations: List[List[float]] = [[] for _ in range(num_cells)]
    per_cell_balance_hits = [0] * num_cells

    for snap in snapshots:
        voltages = snap.voltages[:num_cells]
        pack_mean = statistics.mean(voltages)
        for i, v in enumerate(voltages):
            per_cell_deviations[i].append((v - pack_mean) * 1000.0)  # in mV
        if snap.balancing:
            for i, is_balancing in enumerate(snap.balancing[:num_cells]):
                if is_balancing:
                    per_cell_balance_hits[i] += 1

    avg_deviation_mv = [statistics.mean(devs) for devs in per_cell_deviations]
    avg_voltage = [
        statistics.mean(s.voltages[i] for s in snapshots) for i in range(num_cells)
    ]

    median_dev = statistics.median(avg_deviation_mv)
    abs_diffs = [abs(d - median_dev) for d in avg_deviation_mv]
    mad = statistics.median(abs_diffs) * 1.4826  # scaled MAD ~ robust std-dev estimate

    threshold_mv = max(MIN_ABSOLUTE_MV, MAD_MULTIPLIER * mad)

    cell_stats = []
    for i in range(num_cells):
        is_outlier = abs(avg_deviation_mv[i] - median_dev) >= threshold_mv
        cell_stats.append(
            CellStat(
                index=i + 1,
                mean_voltage=round(avg_voltage[i], 4),
                deviation_mv=round(avg_deviation_mv[i], 1),
                balance_fraction=round(per_cell_balance_hits[i] / len(snapshots), 2),
                is_outlier=is_outlier,
            )
        )

    outliers = [c for c in cell_stats if c.is_outlier]
    flagged_cell = None
    if outliers:
        # Report the single worst offender - the one furthest from the
        # rest of the pack - rather than overwhelming the user with a
        # list when it's usually one cell causing the spread.
        flagged_cell = max(outliers, key=lambda c: abs(c.deviation_mv - median_dev))

    return SessionCellReport(
        num_cells=num_cells,
        cell_stats=cell_stats,
        flagged_cell=flagged_cell,
        num_snapshots=len(snapshots),
    )


# ---------------------------------------------------------------------------
# Cross-session history (local JSON log, keyed by serial port)
# ---------------------------------------------------------------------------

def _log_path_for(port: str) -> str:
    os.makedirs(_LOG_DIR, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", port)
    return os.path.join(_LOG_DIR, f"{safe_name}.json")


def load_history(port: str) -> List[dict]:
    """Returns every past session's summary for this port, oldest
    first. Returns an empty list the first time (no file yet) or if
    the log is corrupt - history is a nice-to-have, so a bad file
    shouldn't block a new reading."""
    path = _log_path_for(port)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def append_session(port: str, report: SessionCellReport, timestamp: float) -> None:
    """Appends this session's result to the port's history file. Keeps
    only the most recent 50 sessions so the file doesn't grow forever."""
    history = load_history(port)
    history.append(
        {
            "timestamp": timestamp,
            "num_cells": report.num_cells,
            "flagged_cell": report.flagged_cell.index if report.flagged_cell else None,
        }
    )
    history = history[-50:]

    path = _log_path_for(port)
    try:
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
    except OSError:
        pass  # non-critical - this session's live result still displays fine


def summarize_history(history: List[dict], lookback: int = 5) -> Optional[dict]:
    """
    Looks at the last `lookback` sessions (including the one just
    appended) and reports which cell has been flagged most often -
    this is the real "does this keep happening" signal that a single
    session can't give you.

    Returns None if there's fewer than 2 sessions of history yet.
    """
    recent = history[-lookback:]
    if len(recent) < 2:
        return None

    flagged_counts: dict = {}
    for entry in recent:
        cell = entry.get("flagged_cell")
        if cell is not None:
            flagged_counts[cell] = flagged_counts.get(cell, 0) + 1

    if not flagged_counts:
        return {"worst_cell": None, "count": 0, "out_of": len(recent)}

    worst_cell = max(flagged_counts, key=flagged_counts.get)
    return {
        "worst_cell": worst_cell,
        "count": flagged_counts[worst_cell],
        "out_of": len(recent),
    }
