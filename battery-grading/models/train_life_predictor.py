"""
Training script: learns from a real lab dataset (114k+ readings, 138
batteries) to build a model that can predict a battery's total cycle
life by looking at ONLY the early cycles (the first 100 cycles).

This is more robust than our earlier simple "capacity ratio" heuristic
because it:
  1. Uses the degradation TRAJECTORY (rate of IR rise, rate of
     capacity fade, variance), not just current capacity
  2. Also factors in confounding factors like temperature and
     charging behavior
  3. Was validated on real degraded-battery data, not hand-picked
     thresholds

The feature engineering approach is inspired by Severson et al.
(Nature Energy, 2019), "Data-driven prediction of battery cycle life
before capacity degradation" - early-cycle IR/capacity trends alone
are enough to predict life, without needing to run the full cycle
life through.
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

RAW_CSV = "raw_data/Lithium-Ion_Battery_Cycle_Life.csv"
EARLY_CYCLE_WINDOW = 100  # only use the first 100 cycles for features
MODEL_OUT = "models/life_predictor.joblib"
META_OUT = "models/life_predictor_meta.json"

FEATURE_NAMES = [
    "ir_at_window_end",
    "ir_slope",
    "qd_at_window_end",
    "qd_slope",
    "qd_variance",
    "tavg_mean",
    "chargetime_mean",
    "qd_fade_ratio",   # % capacity lost by the end of the window
]


def extract_features(df_battery: pd.DataFrame) -> dict:
    """Builds features from a battery's early cycles."""
    window = df_battery[
        (df_battery["cycle"] > 1) & (df_battery["cycle"] <= EARLY_CYCLE_WINDOW)
    ].sort_values("cycle")

    if len(window) < 10:
        return None  # too little data for a reliable feature set

    cycles = window["cycle"].values
    ir = window["IR"].values
    qd = window["QD"].values

    ir_slope = np.polyfit(cycles, ir, 1)[0]
    qd_slope = np.polyfit(cycles, qd, 1)[0]

    qd_start = qd[0]
    qd_end = qd[-1]
    fade_ratio = (qd_start - qd_end) / qd_start if qd_start > 0 else 0.0

    return {
        "ir_at_window_end": ir[-1],
        "ir_slope": ir_slope,
        "qd_at_window_end": qd_end,
        "qd_slope": qd_slope,
        "qd_variance": float(np.var(qd)),
        "tavg_mean": float(window["Tavg"].mean()),
        "chargetime_mean": float(window["chargetime"].mean()),
        "qd_fade_ratio": fade_ratio,
    }


def build_dataset(raw_csv_path: str) -> pd.DataFrame:
    print("Loading raw CSV...")
    df = pd.read_csv(raw_csv_path)
    df = df.dropna(subset=["cycle_life"])  # drop rows with no label

    rows = []
    for battery_id, group in df.groupby("battery_id"):
        feats = extract_features(group)
        if feats is None:
            continue
        feats["battery_id"] = battery_id
        feats["cycle_life"] = group["cycle_life"].iloc[0]
        rows.append(feats)

    return pd.DataFrame(rows)


def main():
    dataset = build_dataset(RAW_CSV)
    print(f"Total usable batteries (features successfully built): {len(dataset)}")

    X = dataset[FEATURE_NAMES]
    y = dataset["cycle_life"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print()
    print("=" * 50)
    print(f"Test set size: {len(X_test)} batteries")
    print(f"Mean Absolute Error: {mae:.1f} cycles")
    print(f"R² score: {r2:.3f}")
    print(f"Average actual cycle_life in test set: {y_test.mean():.1f}")
    print(f"MAE as % of average: {(mae/y_test.mean())*100:.1f}%")
    print("=" * 50)

    print("\nFeature importance:")
    for name, imp in sorted(
        zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {name}: {imp:.3f}")

    # Retrain the final model on all data (more data = more robust for production)
    final_model = RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42
    )
    final_model.fit(X, y)

    joblib.dump(final_model, MODEL_OUT)
    with open(META_OUT, "w") as f:
        json.dump(
            {
                "feature_names": FEATURE_NAMES,
                "early_cycle_window": EARLY_CYCLE_WINDOW,
                "test_mae_cycles": round(mae, 1),
                "test_r2": round(r2, 3),
                "n_training_batteries": len(dataset),
            },
            f,
            indent=2,
        )
    print(f"\nModel saved: {MODEL_OUT}")
    print(f"Metadata saved: {META_OUT}")


if __name__ == "__main__":
    main()
