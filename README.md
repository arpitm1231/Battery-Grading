# 🔋 Battery Grading

**Grade retired EV batteries for safe second-life reuse — powered by real BMS hardware and machine learning.**

Battery Grading reads live data straight from real Battery Management System (BMS) hardware — or a CSV, or a simulator — and turns it into a clear verdict: how healthy is this battery, what grade does it deserve, and what should it be reused for (solar storage, telecom backup, or recycling)?

Built for a real problem: thousands of EV and e-rickshaw batteries are retired every year with no fast, reliable way to tell which ones still have life left. Guessing wrong either wastes good batteries or puts unsafe ones back into service.

---

## ✨ Features

| | |
|---|---|
| 🔌 **Real hardware support** | Native serial parsers for **Daly** and **JBD** BMS protocols — the two most common chipsets in the Indian EV/e-rickshaw aftermarket — plus a JSON-lines mode for any DIY Arduino/ESP32 BMS |
| 🧪 **Works without hardware too** | CSV upload and a built-in simulator mean you can test and demo the full pipeline with zero hardware in hand |
| 📊 **State of Health (SoH)** | Computes real capacity-ratio SoH% and degradation rate from measured data |
| 🤖 **ML-backed life prediction** | A Random Forest model (trained on 138 real battery cycle-life records) predicts remaining cycle life from early-cycle internal resistance and charge-time trends |
| 🛡️ **Safety override** | If the ML model looks too pessimistic against a battery's actual measured SoH, the measured data wins — the app never lets a model override reality |
| 🅰️ **A/B/C grading + use-case** | Every battery gets a grade and a recommended second life: solar storage, telecom backup, or recycling |
| 🔍 **Per-cell diagnostics** | Flags the *one* weak cell to replace instead of scrapping the whole pack, using robust statistical outlier detection (MAD-based) |
| 🗂️ **Cross-session history** | Tracks whether the same cell keeps getting flagged across multiple test sessions — the real signal of degradation, not just one noisy reading |
| 🧩 **Pluggable architecture** | A clean `BatteryDataSource` interface means adding a new hardware protocol never touches the grading logic |

---

## 🚀 Quick start

\```bash
pip install -r requirements.txt
streamlit run app.py
\```

The dashboard opens at `http://localhost:8501`. No hardware needed to try it — pick **Sample data** or **Simulated BMS** from the sidebar to get a full walkthrough immediately.

---

## 🖥️ Three ways to feed it data

| Mode | What it needs | Best for |
|---|---|---|
| **Sample data** | Nothing | Instant demo — 3 pre-generated batteries (healthy / medium / degraded) |
| **CSV upload** | A CSV file | Uploading your own dataset or research-format data |
| **Live BMS (hardware)** | A BMS connected via USB-serial, *or* nothing (simulated sub-mode) | Real-world grading on an actual battery |

### Connecting real BMS hardware

1. Plug your BMS in via USB-to-serial.
2. In the sidebar, choose **Live BMS (hardware)** → **Real hardware (Serial port)**.
3. The app auto-detects available serial ports. Pick the port, baud rate, and protocol.
4. Click **Connect & Read**.

Requires `pyserial` (already in `requirements.txt`). If it's missing, the app shows a friendly warning and disables this mode instead of crashing.

#### Supported BMS protocols

| Protocol | Notes |
|---|---|
| **Daly Smart BMS** (UART/485) | Official Daly protocol (data IDs `0x90`/`0x92`/`0x93`). Doesn't report nameplate capacity over the wire — enter it manually from the battery label. |
| **JBD / Overkill Solar / "LLT"** (UART) | Widely used on Overkill Solar, Xiaoxiang, and many rebranded 4S–24S LiFePO4 boards. Rated capacity is read automatically. |
| **JSON-lines** (DIY / Arduino / ESP32) | For any microcontroller you control. Print one JSON object per line: `{"voltage": 3.72, "current": 1.15, "temperature": 28.4, "cycle_count": 150, "capacity_ah": 95.2, "rated_capacity_ah": 100, "internal_resistance": 0.015}` (`internal_resistance` optional, rest required). |

Both binary protocols were implemented against their publicly documented specs. Exact register layouts can vary slightly across firmware revisions — if readings look off for a specific board, `data_sources/bms_protocols.py` has commented byte offsets that are easy to adjust.

**Not yet supported:** JK-BMS (multiple incompatible protocol versions across firmware) and CAN-bus EV packs (needs `python-can` + a manufacturer DBC file — a different transport entirely). Adding either is a new `BatteryDataSource` implementation; the rest of the app doesn't need to change.

---

## 🧠 How grading works

1. **Basic mode** (always available): thresholds on measured SoH% and degradation rate.
2. **ML mode** (when internal resistance data is available): a Random Forest predicts remaining cycle life.
   - **Test performance:** MAE ±106 cycles, R² = 0.71 (138 batteries, 80/20 split)
   - **Most predictive features:** internal resistance trend and charge time — not just raw capacity
3. **Safety override:** if the ML model says life is over but the battery's *measured* SoH is still high, the measured value is trusted instead — flagged in the UI as **"ML + SAFETY OVERRIDE"**.

\```bash
# Retrain the model
python3 models/train_life_predictor.py
\```

**Known limitation:** the training set is small (138 samples), so the model can underestimate life for long-life batteries — which is exactly what the safety override exists to catch.

---

## 🔍 Per-cell diagnostics

When using **Daly** or **JBD** real-hardware mode, the app polls every individual cell's voltage and flags the one behaving differently from the rest of the pack — so you know which single cell to replace, not the whole pack.

- Each cell's deviation from the pack average is measured across the whole session, not one noisy sample.
- A cell is flagged only if it clears **both** a fixed floor (30 mV) *and* a multiple of the pack's own natural spread (MAD — median absolute deviation). A tight, healthy pack won't trigger false alarms; a naturally wider pack won't have every cell flagged either.
- A consistently **lower**-voltage cell is a strong signal of reduced capacity. A consistently **higher** or frequently rebalanced cell is shown too, but is less conclusive — BMS units balance high cells down as normal operation.
- **Cross-session history:** each session's result is appended to a local log (`cell_health_log/`, one file per port), so repeated flags across sessions ("Cell 7 flagged in 4 of the last 5 sessions") become the real signal — not any single reading.

This section only appears for real hardware sessions (Daly/JBD) or clearly-labeled demo data (Sample/Simulated modes). **An uploaded CSV never gets fabricated per-cell detail** — that would be misleading for someone's real battery.

---

## 📁 Project structure

\```
battery-grading/
├── app.py                        # Streamlit dashboard — entry point
├── data_sources/                 # Pluggable data ingestion
│   ├── base.py                     # Interface every source implements
│   ├── csv_source.py               # Generic CSV upload
│   ├── severson_source.py          # Research-dataset CSV format (with IR)
│   ├── bms_source.py               # Real hardware over Serial
│   ├── bms_protocols.py            # Byte-level parsers: JSON-lines, Daly, JBD
│   ├── synthetic_cells.py          # Demo per-cell voltages
│   └── simulated_source.py         # In-process simulated battery
├── models/
│   ├── soh_estimator.py            # Capacity-ratio SoH% + degradation rate
│   ├── life_predictor.py           # Loads the trained RandomForest model
│   └── train_life_predictor.py     # Training script
├── grading/
│   ├── grader.py                    # Grade A/B/C + use-case logic
│   └── cell_health.py               # Per-cell outlier detection + history log
├── raw_data/                     # 138-cell research dataset
├── sample_data/                  # 3 pre-generated demo batteries
└── tools/
    └── simulate_bms.py           # Standalone script to test real serial hardware
\```

**Key design decision:** `data_sources/base.py` defines a common interface (`connect()`, `read_all()`). Every data source — CSV, simulated, or real hardware — implements the same interface, so the grading logic in `models/` and `grading/` never needs to know where the data came from. Adding a new BMS protocol or vehicle means adding one new class; nothing else in the app changes.

---

## 🛠️ Tech stack

Python · Streamlit · scikit-learn · pandas · pyserial · plotly

---

## 🗺️ Roadmap

- [ ] JK-BMS support
- [ ] CAN-bus EV pack support (`python-can`)
- [ ] Expand ML training set beyond 138 samples
- [ ] Field-test against more real hardware/firmware revisions

---

## 🤝 Contributing / feedback

Open to feedback from anyone working in EV, battery recycling, or second-life energy storage — especially if you can test this against real hardware this hasn't seen yet.
