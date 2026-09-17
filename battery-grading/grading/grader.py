"""
Looks at SoH and degradation rate to assign a battery a Grade,
and suggests the best second-life use-case for it.

Works in two modes:
  1. Basic (fallback) - threshold-based grading using only measured
     SoH% and a simple degradation rate. Works with any data source
     (whether or not it provides internal resistance).
  2. Robust (ML-backed) - if a remaining-cycle-life prediction is
     available from the trained model (via predict_total_cycle_life),
     it's factored in too. This means a battery that currently looks
     fine but whose IR trend is rising fast can correctly get a lower
     grade - something a static capacity-ratio check alone can't catch.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GradeResult:
    grade: str          # "A", "B", or "C"
    label: str          # human-readable summary
    recommended_use: str
    color: str           # hex color, for use in the UI
    method: str = "basic"  # "basic" or "ml" - which logic was used


def assign_grade(
    soh_percent: float,
    degradation_rate: float,
    ml_prediction: Optional[dict] = None,
) -> GradeResult:
    """
    If ml_prediction is provided (a dict from models/life_predictor.py),
    it's factored into grading too - specifically
    "estimated_remaining_cycles", which indicates how much more life
    the battery physically has left (not just how much capacity
    remains).
    """
    if ml_prediction is not None:
        remaining = ml_prediction["estimated_remaining_cycles"]
        return _grade_with_ml(soh_percent, remaining)
    return _grade_basic(soh_percent, degradation_rate)


def _grade_basic(soh_percent: float, degradation_rate: float) -> GradeResult:
    """Simple threshold-based grading - the fallback when IR data isn't available."""
    if soh_percent >= 80 and degradation_rate < 1.5:
        return GradeResult(
            grade="A",
            label="Healthy - low degradation",
            recommended_use="Best fit for solar home storage or UPS backup",
            color="#4ADE80",
            method="basic",
        )
    elif soh_percent >= 60:
        return GradeResult(
            grade="B",
            label="Moderate wear - suitable for stable, steady use",
            recommended_use="Telecom tower backup or low-demand grid storage",
            color="#FBBF24",
            method="basic",
        )
    else:
        return GradeResult(
            grade="C",
            label="Significant degradation - limited second life",
            recommended_use="Send for material recovery/recycling",
            color="#F87171",
            method="basic",
        )


def _grade_with_ml(soh_percent: float, estimated_remaining_cycles: float) -> GradeResult:
    """
    Robust grading - looks at ML-predicted remaining cycle life
    alongside measured SoH%. This is more reliable because it captures
    the degradation trajectory (how fast IR is rising), not just a
    single snapshot capacity number.

    SAFETY NET: because of the small training set (138 batteries), the
    model can sometimes underestimate life for long-life batteries -
    the battery has already survived more cycles than the model's
    predicted total (estimated_remaining <= 0), yet its MEASURED SoH is
    still high. That's a contradiction which points to the model being
    wrong, not the battery. In this case we trust the measured SoH over
    the ML prediction, so a genuinely good battery doesn't get
    mistakenly graded "C".
    """
    if estimated_remaining_cycles <= 0 and soh_percent >= 70:
        return GradeResult(
            grade="A" if soh_percent >= 80 else "B",
            label=(
                f"Measured SoH is {soh_percent}% (healthy), but the ML model "
                "underestimated this battery's life (a limitation of the "
                "small training set) - measured data was prioritized instead"
            ),
            recommended_use=(
                "Solar home storage or UPS backup"
                if soh_percent >= 80
                else "Telecom tower backup or low-demand grid storage"
            ),
            color="#4ADE80" if soh_percent >= 80 else "#FBBF24",
            method="ml-corrected",
        )

    if estimated_remaining_cycles >= 400 and soh_percent >= 75:
        return GradeResult(
            grade="A",
            label=f"Healthy - ~{int(estimated_remaining_cycles)} cycles remaining (ML predicted)",
            recommended_use="Best fit for solar home storage or UPS backup",
            color="#4ADE80",
            method="ml",
        )
    elif estimated_remaining_cycles >= 150:
        return GradeResult(
            grade="B",
            label=f"Moderate wear - ~{int(estimated_remaining_cycles)} cycles remaining (ML predicted)",
            recommended_use="Telecom tower backup or low-demand grid storage",
            color="#FBBF24",
            method="ml",
        )
    else:
        return GradeResult(
            grade="C",
            label=f"Significant degradation - only ~{int(estimated_remaining_cycles)} cycles remaining",
            recommended_use="Send for material recovery/recycling",
            color="#F87171",
            method="ml",
        )
