"""
Battery Second-Life Grading - Streamlit Dashboard

Takes battery readings from a data source (CSV upload or sample
data), estimates SoH, assigns a grade, and shows everything in
one dashboard.
"""

import os
import csv
import tempfile
import time

import plotly.graph_objects as go
import streamlit as st

from data_sources.csv_source import CSVDataSource
from data_sources.severson_source import SeversonCSVDataSource, is_severson_format
from grading.cell_health import analyze_session, append_session, load_history, summarize_history
from grading.grader import assign_grade
from models.life_predictor import is_available as ml_model_available
from models.life_predictor import predict_total_cycle_life
from models.soh_estimator import estimate_degradation_rate, estimate_soh

st.set_page_config(
    page_title="Battery Grading",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

    :root {
        --bg: #14181C;
        --bg-panel: #1B2126;
        --bg-panel-2: #212932;
        --border: #2C3640;
        --text: #E8E6E1;
        --text-dim: #8B96A1;
        --accent: #2DD4BF;
    }

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: var(--bg);
        color: var(--text);
    }

    section[data-testid="stSidebar"] {
        background-color: var(--bg-panel);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    /* Dropdown / select menus (selectbox, multiselect) are rendered by
       Streamlit in a floating "popover" layer that lives outside the
       sidebar's own DOM tree. Because of that, the dark theme rules
       above never reach them, and they can fall back to a white
       background with light/white text - making the options
       unreadable. These rules force the same dark theme onto that
       popover layer explicitly, wherever it is attached. */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] div,
    ul[data-testid="stSelectboxVirtualDropdown"],
    div[data-baseweb="menu"] {
        background-color: var(--bg-panel-2) !important;
        color: var(--text) !important;
    }

    div[data-baseweb="popover"] li,
    ul[data-testid="stSelectboxVirtualDropdown"] li,
    div[data-baseweb="menu"] li {
        background-color: var(--bg-panel-2) !important;
        color: var(--text) !important;
    }

    div[data-baseweb="popover"] li:hover,
    ul[data-testid="stSelectboxVirtualDropdown"] li:hover,
    div[data-baseweb="menu"] li:hover,
    div[data-baseweb="popover"] li[aria-selected="true"],
    ul[data-testid="stSelectboxVirtualDropdown"] li[aria-selected="true"] {
        background-color: var(--bg) !important;
        color: var(--accent) !important;
    }

    /* The closed select box itself (the box you click to open the
       dropdown) also needs an explicit dark background + border, so
       it doesn't default to white before it's opened. */
    div[data-baseweb="select"] > div {
        background-color: var(--bg-panel-2) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }

    /* st.file_uploader (used by the "CSV upload" sidebar option) is
       another Streamlit widget that ships its own light-theme
       styling by default and isn't covered by the sidebar's blanket
       `* { color: var(--text) }` rule, since its dropzone has an
       explicit light background baked in. Without this, it renders
       as a white box - readable-ish but inconsistent, and the
       "Browse files" button and uploaded-file row are especially
       low-contrast. These rules dark-theme it explicitly, the same
       way the popover/dropdown rules above do for selects. */
    section[data-testid="stFileUploaderDropzone"] {
        background-color: var(--bg-panel-2) !important;
        border: 1px dashed var(--border) !important;
    }

    section[data-testid="stFileUploaderDropzone"] * {
        color: var(--text) !important;
    }

    section[data-testid="stFileUploaderDropzone"] small {
        color: var(--text-dim) !important;
    }

    section[data-testid="stFileUploaderDropzone"] button {
        background-color: var(--bg-panel) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }

    /* the row that appears after a file is uploaded (filename, size,
       remove "x" button) */
    div[data-testid="stFileUploaderFile"] {
        background-color: var(--bg-panel-2) !important;
        color: var(--text) !important;
    }

    div[data-testid="stFileUploaderFile"] * {
        color: var(--text) !important;
    }

    div[data-testid="stFileUploaderFile"] small {
        color: var(--text-dim) !important;
    }

    h1, h2, h3 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        color: var(--text);
    }

    .subtitle {
        color: var(--text-dim);
        font-size: 0.95rem;
        margin-top: -0.6rem;
        margin-bottom: 2rem;
    }

    .hero-card {
        background-color: var(--bg-panel);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: 4px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1rem;
    }

    .hero-label {
        color: var(--text-dim);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }

    .hero-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 3rem;
        font-weight: 600;
        color: var(--accent);
        line-height: 1;
    }

    .grade-badge {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.3rem 0.9rem;
        border-radius: 3px;
        margin-top: 0.6rem;
    }

    .info-card {
        background-color: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 1.2rem 1.4rem;
        height: 100%;
    }

    .info-label {
        color: var(--text-dim);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
    }

    .info-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text);
    }

    .recommend-text {
        color: var(--text);
        font-size: 1rem;
        margin-top: 0.3rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar - data source selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Data Source")
    st.markdown(
        "<div class='subtitle'>Choose where battery readings come from</div>",
        unsafe_allow_html=True,
    )

    source_type = st.radio(
        "Source",
        ["Sample data", "CSV upload", "Live BMS (hardware)"],
        label_visibility="collapsed",
    )

    csv_path = None
    uploaded_file = None
    bms_config = None

    if source_type == "Sample data":
        sample_choice = st.selectbox(
            "Sample battery",
            ["Healthy battery", "Medium wear battery", "Degraded battery"],
        )
        sample_map = {
            "Healthy battery": "sample_data/battery_healthy.csv",
            "Medium wear battery": "sample_data/battery_medium.csv",
            "Degraded battery": "sample_data/battery_degraded.csv",
        }
        csv_path = sample_map[sample_choice]

    elif source_type == "CSV upload":
        uploaded_file = st.file_uploader("Battery data CSV", type=["csv"])
        st.markdown(
            "<div class='subtitle'>Both a simple BMS-style CSV and a "
            "research-grade lab dataset (with IR/QC/QD) are "
            "auto-detected.</div>",
            unsafe_allow_html=True,
        )

    else:
        connection_mode = st.radio(
            "Connection type",
            ["Simulated (no hardware needed)", "Real hardware (Serial port)"],
        )

        if connection_mode == "Simulated (no hardware needed)":
            sim_preset = st.selectbox(
                "Simulated battery condition",
                ["Healthy battery", "Medium wear battery", "Degraded battery"],
            )
            sim_clicked = st.button("Generate & Read", use_container_width=True)
            if sim_clicked:
                bms_config = {"mode": "simulated", "preset": sim_preset}

        else:
            from data_sources.bms_source import (
                SerialLibraryMissingError,
                list_serial_ports,
            )

            try:
                available_ports = list_serial_ports()
            except SerialLibraryMissingError:
                available_ports = None
                st.warning(
                    "Real hardware mode needs the **pyserial** package, which "
                    "isn't installed. Run `pip install pyserial`, then restart "
                    "the app. Until then, use **Simulated** mode above to test "
                    "this flow without hardware.",
                    icon="⚠️",
                )

            if available_ports is not None and not available_ports:
                st.info(
                    "No serial port detected. Connect a real BMS/USB adapter "
                    "and refresh the page.",
                    icon="🔌",
                )
            elif available_ports:
                selected_port = st.selectbox("Serial port", available_ports)
                baudrate = st.selectbox(
                    "Baud rate", [9600, 19200, 38400, 57600, 115200], index=0
                )
                protocol_choice = st.selectbox(
                    "BMS protocol",
                    [
                        "JSON-lines (DIY / Arduino / ESP32)",
                        "Daly Smart BMS (UART/485)",
                        "JBD / Overkill Solar / LLT (UART)",
                    ],
                )
                protocol_map = {
                    "JSON-lines (DIY / Arduino / ESP32)": "json",
                    "Daly Smart BMS (UART/485)": "daly",
                    "JBD / Overkill Solar / LLT (UART)": "jbd",
                }
                protocol = protocol_map[protocol_choice]

                rated_capacity_ah = None
                if protocol == "daly":
                    st.markdown(
                        "<div class='subtitle'>The Daly protocol doesn't report "
                        "nameplate capacity, so enter it manually (check the "
                        "battery/cell label).</div>",
                        unsafe_allow_html=True,
                    )
                    rated_capacity_ah = st.number_input(
                        "Rated capacity (Ah)", min_value=1.0, value=100.0, step=1.0
                    )
                elif protocol == "jbd":
                    st.markdown(
                        "<div class='subtitle'>Rated capacity is read from the "
                        "BMS automatically. Only set this if you want to "
                        "override it.</div>",
                        unsafe_allow_html=True,
                    )
                    override_rated = st.checkbox("Override rated capacity")
                    if override_rated:
                        rated_capacity_ah = st.number_input(
                            "Rated capacity (Ah)", min_value=1.0, value=100.0, step=1.0
                        )

                read_seconds = st.slider(
                    "How long to collect data (seconds)", 3, 30, 10
                )
                connect_clicked = st.button("Connect & Read", use_container_width=True)
                if connect_clicked:
                    bms_config = {
                        "mode": "real",
                        "port": selected_port,
                        "baudrate": baudrate,
                        "read_seconds": read_seconds,
                        "protocol": protocol,
                        "rated_capacity_ah": rated_capacity_ah,
                    }

    st.markdown("---")
    st.markdown(
        "<div class='subtitle'>Battery Second-Life Grading Tool<br>"
        "Modular architecture: CSV and real BMS data sources both feed "
        "the same grading pipeline.</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown("## Battery Second-Life Grading")
st.markdown(
    "<div class='subtitle'>Assess the health of retired EV batteries and "
    "get the best second-life use-case for them</div>",
    unsafe_allow_html=True,
)

readings = None
tmp_path_to_clean = None
load_error = None
cell_report = None
cell_history_summary = None
is_demo_cells = False

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    tmp_path_to_clean = tmp_path

    try:
        if is_severson_format(tmp_path):
            temp_source = SeversonCSVDataSource(tmp_path, battery_id="")
            battery_ids = temp_source.list_battery_ids()
            if not battery_ids:
                load_error = "No battery_id found in the file."
            else:
                chosen_id = st.selectbox(
                    f"This is a research-grade dataset ({len(battery_ids)} "
                    "batteries found) — which battery do you want to grade?",
                    battery_ids,
                )
                source = SeversonCSVDataSource(tmp_path, battery_id=chosen_id)
                readings = source.read_all()
                if not readings:
                    load_error = (
                        f"No valid readings found for '{chosen_id}' "
                        "(it may have too few cycles)."
                    )
        else:
            source = CSVDataSource(tmp_path)
            readings = source.read_all()
            if not readings:
                load_error = "The CSV file is empty or has no valid rows."
    except KeyError as e:
        load_error = (
            f"A required column is missing from the CSV: {e}. Expected "
            "columns: timestamp, voltage, current, temperature, "
            "cycle_count, capacity_ah, rated_capacity_ah (or, for the "
            "research format: IR, QC, QD, Tavg, cycle, battery_id)."
        )
    except csv.Error as e:
        load_error = f"Error reading the CSV file: {e}"
    except (ValueError, TypeError) as e:
        load_error = f"The CSV file has invalid data: {e}"
    except Exception as e:
        load_error = f"Unexpected error processing the file: {e}"

elif csv_path is not None:
    try:
        source = CSVDataSource(csv_path)
        readings = source.read_all()

        # Sample data has no real per-cell voltages, so this generates
        # clearly-labeled synthetic demo data just to show what the
        # per-cell diagnostics feature looks like - never done for a
        # user's own uploaded CSV (see synthetic_cells.py).
        from data_sources.synthetic_cells import generate_demo_cell_snapshots

        demo_snapshots = generate_demo_cell_snapshots(condition=sample_choice)
        cell_report = analyze_session(demo_snapshots)
        is_demo_cells = True
    except Exception as e:
        load_error = f"Could not load sample data: {e}"

elif bms_config is not None:
    if bms_config["mode"] == "simulated":
        from data_sources.simulated_source import SimulatedBMSDataSource

        try:
            source = SimulatedBMSDataSource(preset=bms_config["preset"])
            readings = source.read_all()

            if source.cell_snapshots:
                cell_report = analyze_session(source.cell_snapshots)
                is_demo_cells = True
        except Exception as e:
            load_error = f"Error in the simulator: {e}"

    else:
        from data_sources.bms_source import SerialBMSDataSource, SerialLibraryMissingError

        with st.spinner(f"Collecting data from the BMS for {bms_config['read_seconds']}s..."):
            try:
                source = SerialBMSDataSource(
                    port=bms_config["port"],
                    baudrate=bms_config["baudrate"],
                    read_duration_seconds=bms_config["read_seconds"],
                    min_samples=1,
                    protocol=bms_config["protocol"],
                    rated_capacity_ah=bms_config["rated_capacity_ah"],
                )
                readings = source.read_all()

                if source.cell_snapshots:
                    cell_report = analyze_session(source.cell_snapshots)
                    if cell_report is not None:
                        append_session(
                            bms_config["port"], cell_report, time.time()
                        )
                        history = load_history(bms_config["port"])
                        cell_history_summary = summarize_history(history)

                source.close()
            except SerialLibraryMissingError as e:
                load_error = str(e)
            except (ConnectionError, ValueError) as e:
                load_error = str(e)
            except Exception as e:
                load_error = f"Unexpected error while connecting to the BMS: {e}"

if load_error:
    st.error(load_error, icon="⚠️")

if tmp_path_to_clean and os.path.exists(tmp_path_to_clean):
    try:
        os.unlink(tmp_path_to_clean)
    except OSError:
        pass  # cleanup failure isn't critical

if readings:
    try:
        soh = estimate_soh(readings)
        degradation_rate = estimate_degradation_rate(readings)
    except (ValueError, ZeroDivisionError) as e:
        st.error(f"Could not calculate SoH: {e}", icon="⚠️")
        st.stop()

    ml_prediction = None
    if ml_model_available():
        try:
            ml_prediction = predict_total_cycle_life(readings)
        except Exception:
            # If the ML prediction fails, basic grading still runs -
            # that's the whole point of this fallback (robustness)
            ml_prediction = None

    result = assign_grade(soh, degradation_rate, ml_prediction=ml_prediction)

    _badge_style = (
        "font-family:'IBM Plex Mono',monospace; font-size:0.7rem; "
        "padding:0.15rem 0.5rem; border-radius:3px; margin-left:0.6rem;"
    )
    if result.method == "ml":
        method_badge = f"<span style='{_badge_style} color:#2DD4BF; border:1px solid #2DD4BF55;'>ML MODEL</span>"
    elif result.method == "ml-corrected":
        method_badge = f"<span style='{_badge_style} color:#FBBF24; border:1px solid #FBBF2455;'>ML + SAFETY OVERRIDE</span>"
    else:
        method_badge = f"<span style='{_badge_style} color:#8B96A1; border:1px solid #2C3640;'>BASIC ESTIMATE</span>"

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown(
            f"""
            <div class="hero-card">
                <div class="hero-label">State of Health</div>
                <div class="hero-value">{soh}%</div>
                <div class="grade-badge" style="background-color:{result.color}22; color:{result.color}; border:1px solid {result.color}55;">
                    GRADE {result.grade}
                </div>{method_badge}
                <div class="recommend-text" style="margin-top:0.6rem;">{result.label}</div>
                <div class="recommend-text">→ {result.recommended_use}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">Cycle Count</div>
                    <div class="info-value">{readings[-1].cycle_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with subcol2:
            label = "Fade / 100 cycles" if not ml_prediction else "Predicted Remaining"
            value = (
                f"{degradation_rate}%"
                if not ml_prediction
                else f"{int(ml_prediction['estimated_remaining_cycles'])} cyc"
            )
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">{label}</div>
                    <div class="info-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if ml_prediction:
            st.markdown(
                f"""
                <div class="info-card" style="margin-top:1rem;">
                    <div class="info-label">Model confidence</div>
                    <div class="recommend-text" style="margin-top:0.3rem;">
                        Trained on 138 real batteries · Test MAE ±{ml_prediction['model_test_mae']:.0f} cycles ·
                        R² {ml_prediction['model_r2']:.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Degradation chart
    cycles = [r.cycle_count for r in readings]
    capacity_pct = [
        round((r.capacity_ah / r.rated_capacity_ah) * 100, 2) for r in readings
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cycles,
            y=capacity_pct,
            mode="lines+markers",
            line=dict(color="#2DD4BF", width=2),
            marker=dict(size=5, color="#2DD4BF"),
            name="Capacity retained",
        )
    )
    fig.add_hline(y=80, line_dash="dot", line_color="#4ADE80", opacity=0.5)
    fig.add_hline(y=60, line_dash="dot", line_color="#FBBF24", opacity=0.5)

    fig.update_layout(
        title="Capacity Retention vs Cycle Count",
        xaxis_title="Cycle count",
        yaxis_title="Capacity retained (%)",
        plot_bgcolor="#1B2126",
        paper_bgcolor="#1B2126",
        font=dict(family="IBM Plex Sans", color="#E8E6E1"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=380,
        xaxis=dict(gridcolor="#2C3640"),
        yaxis=dict(gridcolor="#2C3640"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------
    # Per-cell diagnostics - real per-cell data for Daly/JBD hardware
    # sessions, or clearly-labeled synthetic demo data for Sample
    # data / Simulated BMS modes (which have no real per-cell data).
    # -----------------------------------------------------------------
    if cell_report is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        header = "### Per-Cell Diagnostics"
        if is_demo_cells:
            header += "  🧪 *(simulated demo data, not from real hardware)*"
        st.markdown(header)

        if cell_report.flagged_cell is not None:
            fc = cell_report.flagged_cell
            direction = "lower" if fc.deviation_mv < 0 else "higher"
            st.markdown(
                f"""
                <div class="hero-card" style="border-left-color:#F87171;">
                    <div class="hero-label">Suspect Cell</div>
                    <div class="hero-value" style="font-size:2rem; color:#F87171;">Cell {fc.index}</div>
                    <div class="recommend-text">
                        Runs {abs(fc.deviation_mv):.0f}mV {direction} than the pack average
                        across {cell_report.num_snapshots} readings this session.
                        A consistently lower cell is the stronger signal of a
                        failing cell (lower usable capacity); a consistently
                        higher or frequently-balanced cell is worth
                        monitoring but is less conclusive on its own.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="hero-card" style="border-left-color:#4ADE80;">
                    <div class="recommend-text">✅ All cells are within normal spread of each other this session - no single outlier cell detected.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if is_demo_cells:
            st.markdown(
                "<div class='subtitle'>This is simulated data meant to show "
                "what the diagnostic looks like - it doesn't reflect a real "
                "battery, and history isn't tracked across sessions for demo "
                "data. Switch to real hardware (Daly/JBD) mode for an actual "
                "diagnosis and cross-session trend tracking.</div>",
                unsafe_allow_html=True,
            )
        elif cell_history_summary is not None and cell_history_summary["worst_cell"] is not None:
            st.markdown(
                f"<div class='subtitle'>History on this port: Cell "
                f"{cell_history_summary['worst_cell']} has been flagged in "
                f"{cell_history_summary['count']} of the last "
                f"{cell_history_summary['out_of']} sessions - "
                f"{'a recurring pattern worth investigating' if cell_history_summary['count'] >= cell_history_summary['out_of'] * 0.5 else 'keep an eye on it over more sessions'}."
                f"</div>",
                unsafe_allow_html=True,
            )
        elif cell_history_summary is None:
            st.markdown(
                "<div class='subtitle'>Not enough history on this port yet to "
                "show a trend - run a few more sessions to see if the same "
                "cell keeps showing up.</div>",
                unsafe_allow_html=True,
            )

        cell_fig = go.Figure()
        bar_colors = [
            "#F87171" if (cell_report.flagged_cell and c.index == cell_report.flagged_cell.index) else "#2DD4BF"
            for c in cell_report.cell_stats
        ]
        cell_fig.add_trace(
            go.Bar(
                x=[f"Cell {c.index}" for c in cell_report.cell_stats],
                y=[c.mean_voltage for c in cell_report.cell_stats],
                marker_color=bar_colors,
            )
        )
        pack_avg = sum(c.mean_voltage for c in cell_report.cell_stats) / len(cell_report.cell_stats)
        cell_fig.add_hline(y=pack_avg, line_dash="dot", line_color="#8B96A1", opacity=0.6)
        cell_fig.update_layout(
            title="Cell Voltages (this session's average)",
            yaxis_title="Voltage (V)",
            plot_bgcolor="#1B2126",
            paper_bgcolor="#1B2126",
            font=dict(family="IBM Plex Sans", color="#E8E6E1"),
            margin=dict(l=40, r=20, t=50, b=40),
            height=320,
            xaxis=dict(gridcolor="#2C3640"),
            yaxis=dict(gridcolor="#2C3640"),
            showlegend=False,
        )
        st.plotly_chart(cell_fig, use_container_width=True)

    with st.expander("View raw readings"):
        st.dataframe(
            [
                {
                    "Timestamp": r.timestamp,
                    "Voltage (V)": r.voltage,
                    "Current (A)": r.current,
                    "Temp (°C)": r.temperature,
                    "Cycle": r.cycle_count,
                    "Capacity (Ah)": r.capacity_ah,
                }
                for r in readings
            ],
            use_container_width=True,
        )

else:
    st.markdown(
        """
        <div class="hero-card" style="text-align:center; padding: 3rem;">
            <div style="font-size: 2rem;">🔋</div>
            <div class="subtitle" style="margin-top: 1rem;">
                Choose a data source in the sidebar to get started
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
