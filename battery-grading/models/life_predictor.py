"""
Uses a trained ML model (trained on a real dataset) to predict a
battery's total/remaining cycle life.

Graceful fallback: if readings don't include internal_resistance
(e.g. from a basic voltage/current-only BMS), this returns None and
the grading logic automatically falls back to the simple
capacity-ratio approach - no crash.
"""

import json
import os
from typing import List, Optional

from data_sources.base import BatteryReading

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "life_predictor.joblib")
_META_PATH = os.path.join(os.path.dirname(__file__), "life_predictor_meta.json")

_model = None
_meta = None
_deps_missing = False


def _load_model():
    global _model, _meta, _deps_missing
    if _model is None:
        import joblib  # lazy import - so if sklearn/joblib aren't
        # installed, the ML feature disables gracefully instead of
        # crashing the whole app
        _model = joblib.load(_MODEL_PATH)
        with open(_META_PATH) as f:
            _meta = json.load(f)
    return _model, _meta


def is_available() -> bool:
    if _deps_missing:
        return False
    if not os.path.exists(_MODEL_PATH):
        return False
    try:
        import joblib  # noqa: F401
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


def predict_total_cycle_life(readings: List[BatteryReading]) -> Optional[dict]:
    """
    Builds features from the readings (if internal_resistance is
    available) and predicts total cycle life using the trained model.

    Returns None if a prediction isn't possible (IR data missing, not
    enough cycles of data, or the ML libraries aren't installed).
    """
    if not is_available():
        return None

    import numpy as np

    try:
        model, meta = _load_model()
    except Exception:
        # Model file corrupt/missing/incompatible - don't crash,
        # just fall back to basic grading
        global _deps_missing
        _deps_missing = True
        return None
    window = meta["early_cycle_window"]

    usable = [
        r for r in readings
        if r.cycle_count > 1 and r.cycle_count <= window and r.internal_resistance is not None
    ]

    if len(usable) < 10:
        return None  # not enough IR data / early-cycle data

    usable.sort(key=lambda r: r.cycle_count)
    cycles = np.array([r.cycle_count for r in usable])
    ir = np.array([r.internal_resistance for r in usable])
    qd = np.array([r.capacity_ah for r in usable])
    temps = np.array([r.temperature for r in usable])
    # chargetime isn't directly available in BatteryReading - if it's
    # added in the future, use it here; for now we default to 0 (the
    # model still predicts reasonably without this feature, just with
    # slightly less confidence)

    ir_slope = np.polyfit(cycles, ir, 1)[0]
    qd_slope = np.polyfit(cycles, qd, 1)[0]
    qd_start, qd_end = qd[0], qd[-1]
    fade_ratio = (qd_start - qd_end) / qd_start if qd_start > 0 else 0.0

    features = np.array([[
        ir[-1],           # ir_at_window_end
        ir_slope,         # ir_slope
        qd_end,            # qd_at_window_end
        qd_slope,          # qd_slope
        float(np.var(qd)), # qd_variance
        float(temps.mean()),  # tavg_mean
        0.0,               # chargetime_mean (not tracked in BatteryReading yet)
        fade_ratio,         # qd_fade_ratio
    ]])

    predicted_total_life = float(model.predict(features)[0])
    current_cycle = readings[-1].cycle_count
    remaining = max(0.0, predicted_total_life - current_cycle)

    return {
        "predicted_total_cycle_life": round(predicted_total_life, 0),
        "estimated_remaining_cycles": round(remaining, 0),
        "current_cycle": current_cycle,
        "model_test_mae": meta["test_mae_cycles"],
        "model_r2": meta["test_r2"],
    }
