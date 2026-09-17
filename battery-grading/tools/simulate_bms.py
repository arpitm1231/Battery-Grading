"""
Simulator for testing SerialBMSDataSource without real BMS hardware.
This creates a virtual serial port pair (pty) - it writes fake battery
data on one end, and the other end can be given to our
SerialBMSDataSource as if it were real hardware.

Run it with:
    python3 tools/simulate_bms.py

This will print a port path to the console (like /dev/pts/5) - use
that in app.py's "Live BMS" option for testing.
"""

import json
import os
import pty
import random
import time


def start_fake_bms(cycle_count: int = 250, rated_capacity: float = 100.0):
    master_fd, slave_fd = pty.openpty()
    slave_name = os.ttyname(slave_fd)

    print(f"Fake BMS running. Connect to: {slave_name}")
    print("Press Ctrl+C to stop.\n")

    # Simulate a slightly degraded battery - 82% SoH
    current_capacity = rated_capacity * 0.82
    ir = 0.018

    try:
        while True:
            reading = {
                "voltage": round(random.uniform(3.6, 4.1), 3),
                "current": round(random.uniform(-2.0, 2.0), 3),
                "temperature": round(random.uniform(28, 38), 1),
                "cycle_count": cycle_count,
                "capacity_ah": round(current_capacity + random.uniform(-0.1, 0.1), 3),
                "rated_capacity_ah": rated_capacity,
                "internal_resistance": round(ir + random.uniform(-0.001, 0.001), 4),
            }
            line = (json.dumps(reading) + "\n").encode("utf-8")
            os.write(master_fd, line)
            print("→", reading)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nSimulator stopped.")
    finally:
        os.close(master_fd)
        os.close(slave_fd)


if __name__ == "__main__":
    start_fake_bms()
