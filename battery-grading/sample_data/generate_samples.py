"""
Generates realistic sample battery datasets for testing, with 3
different health conditions (good, medium, poor battery) so the
grading logic can be properly tested.
"""

import csv
import random


def generate_battery_csv(
    file_path: str,
    rated_capacity_ah: float,
    total_cycles: int,
    fade_per_100_cycles: float,
    readings_per_run: int = 20,
):
    """Writes one battery's simulated data to a CSV file."""
    rows = []
    timestamp = 0.0

    for i in range(readings_per_run):
        cycle = int(total_cycles * (i + 1) / readings_per_run)
        fade_pct = (cycle / 100) * fade_per_100_cycles
        capacity = rated_capacity_ah * (1 - fade_pct / 100)
        capacity += random.uniform(-0.05, 0.05)  # a bit of measurement noise

        rows.append(
            {
                "timestamp": round(timestamp, 1),
                "voltage": round(random.uniform(3.6, 4.1), 3),
                "current": round(random.uniform(-2.0, 2.0), 3),
                "temperature": round(random.uniform(25, 40), 1),
                "cycle_count": cycle,
                "capacity_ah": round(max(0.1, capacity), 3),
                "rated_capacity_ah": rated_capacity_ah,
            }
        )
        timestamp += 300  # one reading every 5 minutes

    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    random.seed(42)

    # Healthy battery - Grade A expected (fewer cycles, low fade)
    generate_battery_csv(
        "sample_data/battery_healthy.csv",
        rated_capacity_ah=100,
        total_cycles=300,
        fade_per_100_cycles=1.0,
    )

    # Medium wear battery - Grade B expected
    generate_battery_csv(
        "sample_data/battery_medium.csv",
        rated_capacity_ah=100,
        total_cycles=800,
        fade_per_100_cycles=3.0,
    )

    # Heavily degraded battery - Grade C expected
    generate_battery_csv(
        "sample_data/battery_degraded.csv",
        rated_capacity_ah=100,
        total_cycles=1500,
        fade_per_100_cycles=5.5,
    )

    print("3 sample battery CSVs generated in sample_data/")
