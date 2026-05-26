"""app.py — Streamlit Dashboard for GeneralSim

Run with:
    streamlit run app.py
"""

import io
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ── ensure workspace is on the path ─────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from scenario import Radar, Target, Scenario
from signal_gen import WaveformConfig, NoiseConfig, SignalGenerator
from processing import ProcessingConfig, WindowType, RadarProcessor
from fusion import associate_and_fuse, FusedTarget
from interceptor import InterceptorSystem, InterceptBlackbox
from feature_extraction import FEATURE_NAMES, extract_from_system_target

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def fig_to_st(fig) -> None:
    """Render a matplotlib figure inside Streamlit."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


def plot_range_doppler(results, radar_idx: int):
    """Create a 2-panel range-Doppler / CFAR figure and return it."""
    rd_map = results["rd_map"]
    detection_mask = results["detection_mask"]
    range_axis = results["range_axis"]
    velocity_axis = results["velocity_axis"]
    estimated_targets = results["estimated_targets"]

    rd_dB = 20 * np.log10(np.abs(rd_map) + 1e-30)
    vmax = rd_dB.max()
    vmin = vmax - 60

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Range-Doppler map
    im = axes[0].imshow(
        rd_dB.T, aspect="auto",
        extent=[range_axis[0], range_axis[-1], velocity_axis[0], velocity_axis[-1]],
        origin="lower", cmap="jet", vmin=vmin, vmax=vmax,
    )
    plt.colorbar(im, ax=axes[0], label="dB")
    axes[0].set_xlabel("Range (m)"); axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title(f"Radar {radar_idx} — Range-Doppler Map")

    # CFAR detections
    axes[1].imshow(
        rd_dB.T, aspect="auto",
        extent=[range_axis[0], range_axis[-1], velocity_axis[0], velocity_axis[-1]],
        origin="lower", cmap="jet", vmin=vmin, vmax=vmax, alpha=0.6,
    )
    det_r = [range_axis[np.clip(i, 0, len(range_axis)-1)]
             for i in np.where(detection_mask)[0]]
    det_v = [velocity_axis[np.clip(j, 0, len(velocity_axis)-1)]
             for j in np.where(detection_mask)[1]]
    axes[1].scatter(det_r, det_v, c="yellow", s=6, alpha=0.5, label="CFAR cells")
    for t in estimated_targets:
        axes[1].plot(t["range"], t["velocity"], "r*", markersize=12)
        axes[1].annotate(f"  {t['range']:.0f}m\n  {t['velocity']:.1f}m/s",
                         (t["range"], t["velocity"]), color="white", fontsize=7)
    axes[1].set_xlabel("Range (m)"); axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_title(f"Radar {radar_idx} — CFAR Detections")
    axes[1].legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    return fig


def plot_intercept_heatmap(systems, fused_targets, P, title):
    """Return a heatmap figure for the probability matrix P."""
    n_sys, n_tgt = P.shape
    fig, ax = plt.subplots(figsize=(max(5, n_tgt * 1.4), max(3, n_sys * 0.9)))
    im = ax.imshow(P * 100, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(n_tgt))
    ax.set_xticklabels([f"FT{j}" for j in range(n_tgt)], fontsize=9)
    ax.set_yticks(range(n_sys))
    ax.set_yticklabels([s.name for s in systems], fontsize=9)
    ax.set_title(title)
    for i in range(n_sys):
        for j in range(n_tgt):
            val = P[i, j] * 100
            c = "white" if val < 30 or val > 70 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=9, color=c)
    plt.colorbar(im, ax=ax, label="P(intercept) %")
    plt.tight_layout()
    return fig


def plot_comparison_scatter(P_ana, P_ml):
    """Return a scatter plot comparing analytical vs ML probabilities."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(P_ana.ravel() * 100, P_ml.ravel() * 100,
               c="steelblue", s=80, edgecolors="navy", alpha=0.8)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.4, label="y = x")
    ax.set_xlabel("Analytical P(intercept) [%]")
    ax.set_ylabel("ML P(intercept) [%]")
    ax.set_title("Analytical vs ML")
    ax.set_xlim(-5, 105); ax.set_ylim(-5, 105)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_scenario_figure(scenario, geometry, systems):
    """Thin wrapper: run Scenario.plot_scenario and return the figure."""
    fig, _ax = scenario.plot_scenario(geometry, interceptor_systems=systems,
                                      save_path=None)
    return fig


def apply_consistent_limits(ax, radars, targets, systems, margin=0.2):
    """Set fixed axis limits based on all entities so the plot size stays stable."""
    xs, ys = [], []
    for r in radars:
        xs.append(r.position[0]); ys.append(r.position[1])
    for t in targets:
        xs.append(t.position[0]); ys.append(t.position[1])
    for s in systems:
        xs.append(s.position[0]); ys.append(s.position[1])
        xs.append(s.position[0] + s.max_range)
        xs.append(s.position[0] - s.max_range)
        ys.append(s.position[1] + s.max_range)
        ys.append(s.position[1] - s.max_range)
    if not xs:
        return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max((x_max - x_min) * margin, 1000)
    y_pad = max((y_max - y_min) * margin, 1000)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def draw_border_line(ax, radars, targets, systems):
    """Draw an auto-computed curved border line between defense assets and targets."""
    # Collect X positions of defensive assets (radars + systems)
    defense_xs = [r.position[0] for r in radars] + [s.position[0] for s in systems]
    # Collect X positions of targets
    target_xs = [t.position[0] for t in targets]

    if not defense_xs or not target_xs:
        return

    max_defense_x = max(defense_xs)
    min_target_x = min(target_xs)
    border_x = (max_defense_x + min_target_x) / 2.0

    # Use the full Y extent of the axes so the line spans edge to edge
    y_lo, y_hi = ax.get_ylim()
    y_span = np.linspace(y_lo, y_hi, 200)
    y_min = y_lo
    y_max = y_hi

    # Sinusoidal perturbation for visual appeal
    amplitude = (y_max - y_min) * 0.02
    freq = 3.0 * np.pi / (y_max - y_min + 1e-6)
    border_curve_x = border_x + amplitude * np.sin(freq * (y_span - y_min))

    ax.plot(border_curve_x, y_span, '--', color='#8B0000', alpha=0.55,
            linewidth=1.8, zorder=1)


# ════════════════════════════════════════════════════════════════════
# Page config
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="GeneralSim Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📡 GeneralSim — Radar Simulation Dashboard")
st.caption("Configure radars, targets, and interceptor systems in the sidebar, then press **Run Simulation**.")

# ════════════════════════════════════════════════════════════════════
# Sidebar — Configuration
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Configuration")

    # ── Radars ──────────────────────────────────────────────────────
    st.subheader("Radars")
    n_radars = st.number_input("Number of radars", min_value=1, max_value=12,
                               value=3, step=1)

    radar_defaults = [
        dict(x=0,      y=0,      z=0, fc=10.0, power=60, gain=30, bw=5.0, az=10.0,  el=0.0),
        dict(x=0,      y=15000,  z=0, fc=9.0,  power=58, gain=30, bw=8.0, az=-25.0, el=0.0),
        dict(x=-10000, y=7500,   z=0, fc=10.0, power=60, gain=30, bw=8.0, az=0.0,   el=0.0),
        dict(x=5000,   y=25000,  z=0, fc=9.5,  power=55, gain=28, bw=6.0, az=45.0,  el=0.0),
        dict(x=15000,  y=10000,  z=0, fc=10.5, power=62, gain=32, bw=4.0, az=-10.0, el=0.0),
        dict(x=-5000,  y=20000,  z=0, fc=9.3,  power=57, gain=29, bw=7.0, az=20.0,  el=0.0),
        dict(x=10000,  y=0,      z=0, fc=10.2, power=61, gain=31, bw=5.5, az=-30.0, el=0.0),
        dict(x=20000,  y=15000,  z=0, fc=9.7,  power=59, gain=30, bw=6.5, az=15.0,  el=0.0),
        dict(x=-15000, y=25000,  z=0, fc=10.8, power=63, gain=33, bw=4.5, az=-5.0,  el=0.0),
        dict(x=25000,  y=5000,   z=0, fc=9.1,  power=56, gain=27, bw=9.0, az=60.0,  el=0.0),
        dict(x=0,      y=30000,  z=0, fc=10.4, power=60, gain=30, bw=5.0, az=-45.0, el=0.0),
        dict(x=-20000, y=10000,  z=0, fc=9.6,  power=58, gain=29, bw=7.5, az=30.0,  el=0.0),
    ]

    radar_cfgs = []
    for i in range(n_radars):
        d = radar_defaults[i] if i < len(radar_defaults) else radar_defaults[-1]
        with st.expander(f"Radar {i}", expanded=(i == 0)):
            c1, c2, c3 = st.columns(3)
            rx = c1.number_input(f"X (m)##r{i}x", value=float(d["x"]), step=500.0, key=f"r{i}x")
            ry = c2.number_input(f"Y (m)##r{i}y", value=float(d["y"]), step=500.0, key=f"r{i}y")
            rz = c3.number_input(f"Z (m)##r{i}z", value=float(d["z"]), step=10.0, key=f"r{i}z")
            fc = st.slider(f"Frequency (GHz)##r{i}", 1.0, 20.0,
                           value=float(d["fc"]), step=0.5, key=f"r{i}fc")
            pwr = st.slider(f"TX Power (dBm)##r{i}", 30, 80,
                            value=int(d["power"]), key=f"r{i}pwr")
            gain = st.slider(f"Antenna Gain (dB)##r{i}", 10, 50,
                             value=int(d["gain"]), key=f"r{i}gain")
            bw = st.slider(f"Beamwidth (°)##r{i}", 1.0, 30.0,
                           value=float(d["bw"]), step=0.5, key=f"r{i}bw")
            az = st.slider(f"Look Azimuth (°)##r{i}", -180.0, 180.0,
                           value=float(d["az"]), step=1.0, key=f"r{i}az")
            el = st.slider(f"Look Elevation (°)##r{i}", -45.0, 45.0,
                           value=float(d["el"]), step=1.0, key=f"r{i}el")
            radar_cfgs.append(dict(x=rx, y=ry, z=rz, fc=fc * 1e9,
                                   power=pwr, gain=gain, bw=bw, az=az, el=el))

    # ── Targets ─────────────────────────────────────────────────────
    st.subheader("Targets")
    n_targets = st.number_input("Number of targets", min_value=1, max_value=20,
                                value=5, step=1)

    target_defaults = [
        dict(x=25000,  y=4000,   z=0,   vx=-200, vy=-30,  vz=0, rcs=10),  # inbound from ~NE
        dict(x=15000,  y=1000,   z=500, vx=-150, vy=-10,  vz=0, rcs=5),   # inbound from E
        dict(x=35000,  y=7500,   z=0,   vx=-245, vy=-50,  vz=0, rcs=15),  # inbound from NE
        dict(x=10000,  y=500,    z=0,   vx=-120, vy=-5,   vz=0, rcs=0),   # inbound from E
        dict(x=30000,  y=-2500,  z=0,   vx=-220, vy=20,   vz=0, rcs=8),   # inbound from SE
        dict(x=20000,  y=15000,  z=500, vx=-145, vy=-110, vz=0, rcs=5),   # inbound from NE diagonal
        dict(x=40000,  y=2500,   z=0,   vx=-280, vy=-15,  vz=0, rcs=12),  # inbound from far E
        dict(x=7500,   y=12500,  z=500, vx=-65,  vy=-110, vz=0, rcs=3),   # inbound from N
        dict(x=45000,  y=8000,   z=0,   vx=-305, vy=-55,  vz=0, rcs=9),   # inbound from far NE
        dict(x=18000,  y=-5000,  z=200, vx=-175, vy=50,   vz=0, rcs=7),   # inbound from SE
        dict(x=50000,  y=20000,  z=0,   vx=-280, vy=-110, vz=0, rcs=11),  # inbound from far NE
        dict(x=12000,  y=18000,  z=300, vx=-80,  vy=-125, vz=0, rcs=6),   # inbound from N
        dict(x=38000,  y=-8000,  z=0,   vx=-245, vy=50,   vz=0, rcs=14),  # inbound from far SE
        dict(x=28000,  y=22000,  z=100, vx=-160, vy=-125, vz=0, rcs=4),   # inbound from NE
        dict(x=55000,  y=3000,   z=0,   vx=-330, vy=-20,  vz=0, rcs=16),  # inbound from far E
        dict(x=8000,   y=30000,  z=400, vx=-40,  vy=-155, vz=0, rcs=2),   # inbound from far N
        dict(x=42000,  y=12000,  z=0,   vx=-260, vy=-75,  vz=0, rcs=13),  # inbound from NE
        dict(x=22000,  y=-10000, z=0,   vx=-200, vy=90,   vz=0, rcs=8),   # inbound from SE
        dict(x=60000,  y=25000,  z=200, vx=-295, vy=-125, vz=0, rcs=10),  # inbound from far NE
        dict(x=16000,  y=35000,  z=500, vx=-70,  vy=-155, vz=0, rcs=5),   # inbound from far N
    ]

    target_cfgs = []
    for i in range(n_targets):
        d = target_defaults[i] if i < len(target_defaults) else target_defaults[-1]
        with st.expander(f"Target {i}", expanded=False):
            c1, c2, c3 = st.columns(3)
            tx = c1.number_input(f"X (m)##t{i}x", value=float(d["x"]), step=500.0, key=f"t{i}x")
            ty = c2.number_input(f"Y (m)##t{i}y", value=float(d["y"]), step=500.0, key=f"t{i}y")
            tz = c3.number_input(f"Z (m)##t{i}z", value=float(d["z"]), step=10.0, key=f"t{i}z")
            c4, c5, c6 = st.columns(3)
            vx = c4.number_input(f"Vx (m/s)##t{i}vx", value=float(d["vx"]), step=5.0, key=f"t{i}vx")
            vy = c5.number_input(f"Vy (m/s)##t{i}vy", value=float(d["vy"]), step=5.0, key=f"t{i}vy")
            vz = c6.number_input(f"Vz (m/s)##t{i}vz", value=float(d["vz"]), step=5.0, key=f"t{i}vz")
            rcs = st.slider(f"RCS (dBsm)##t{i}", -20, 30,
                            value=int(d["rcs"]), key=f"t{i}rcs")
            target_cfgs.append(dict(x=tx, y=ty, z=tz,
                                    vx=vx, vy=vy, vz=vz, rcs=rcs))

    # ── Interceptor Systems ─────────────────────────────────────────
    st.subheader("Interceptor Systems")
    n_systems = st.number_input("Number of systems", min_value=1, max_value=12,
                                value=3, step=1)

    sys_defaults = [
        dict(name="SAM Alpha",   x=5000,   y=5000,  z=0, maxr=8000,  minr=500, maxv=400, rt=5.0, sv=2),
        dict(name="SAM Beta",    x=-5000,  y=12500, z=0, maxr=12000, minr=300, maxv=600, rt=3.0, sv=3),
        dict(name="CIWS Delta",  x=2500,   y=2500,  z=0, maxr=2000,  minr=100, maxv=800, rt=1.0, sv=4),
        dict(name="SAM Gamma",   x=10000,  y=20000, z=0, maxr=10000, minr=400, maxv=500, rt=4.0, sv=2),
        dict(name="SAM Epsilon", x=-2500,  y=17500, z=0, maxr=15000, minr=600, maxv=700, rt=6.0, sv=1),
        dict(name="CIWS Zeta",   x=7500,   y=-2500, z=0, maxr=1500,  minr=100, maxv=900, rt=0.5, sv=6),
        dict(name="SAM Eta",     x=20000,  y=8000,  z=0, maxr=20000, minr=500, maxv=550, rt=4.5, sv=2),
        dict(name="SAM Theta",   x=-12000, y=22000, z=0, maxr=18000, minr=400, maxv=650, rt=5.5, sv=3),
        dict(name="CIWS Iota",   x=15000,  y=30000, z=0, maxr=2500,  minr=150, maxv=850, rt=0.8, sv=5),
        dict(name="SAM Kappa",   x=30000,  y=15000, z=0, maxr=25000, minr=700, maxv=600, rt=7.0, sv=1),
        dict(name="SAM Lambda",  x=-8000,  y=5000,  z=0, maxr=14000, minr=350, maxv=480, rt=3.5, sv=3),
        dict(name="CIWS Mu",     x=12000,  y=-5000, z=0, maxr=1800,  minr=100, maxv=950, rt=0.6, sv=6),
    ]

    system_cfgs = []
    for i in range(n_systems):
        d = sys_defaults[i] if i < len(sys_defaults) else sys_defaults[-1]
        with st.expander(f"{d['name']}", expanded=False):
            sname = st.text_input(f"Name##s{i}", value=d["name"], key=f"s{i}name")
            c1, c2, c3 = st.columns(3)
            sx = c1.number_input(f"X (m)##s{i}x", value=float(d["x"]), step=500.0, key=f"s{i}x")
            sy = c2.number_input(f"Y (m)##s{i}y", value=float(d["y"]), step=500.0, key=f"s{i}y")
            sz = c3.number_input(f"Z (m)##s{i}z", value=float(d["z"]), step=10.0, key=f"s{i}z")
            maxr = st.slider(f"Max Range (m)##s{i}", 500, 30000,
                             value=int(d["maxr"]), step=500, key=f"s{i}maxr")
            minr = st.slider(f"Min Range (m)##s{i}", 50, 2000,
                             value=int(d["minr"]), step=50, key=f"s{i}minr")
            maxv = st.slider(f"Max Target Velocity (m/s)##s{i}", 50, 1500,
                             value=int(d["maxv"]), step=50, key=f"s{i}maxv")
            c4, c5 = st.columns(2)
            rt = c4.number_input(f"Reaction Time (s)##s{i}", value=float(d["rt"]),
                                 step=0.5, min_value=0.1, key=f"s{i}rt")
            sv = c5.number_input(f"Salvo Size##s{i}", value=int(d["sv"]),
                                 step=1, min_value=1, max_value=10, key=f"s{i}sv")
            system_cfgs.append(dict(name=sname, x=sx, y=sy, z=sz,
                                    maxr=maxr, minr=minr, maxv=maxv, rt=rt, sv=int(sv)))

    # ── Processing Options ──────────────────────────────────────────
    st.subheader("Processing Options")
    use_clutter = st.checkbox("Enable Clutter", value=True)
    use_mti = st.checkbox("Enable MTI Filter", value=True)
    use_ml = st.checkbox("Use ML (XGBoost) Model", value=True,
                         help="Requires xgb_intercept_model.json to exist.")

    # ── Time-Stepping ───────────────────────────────────────────────
    st.subheader("Time Stepping")
    sim_duration = st.number_input("Duration (s)", min_value=0.0, max_value=600.0,
                                   value=60.0, step=10.0,
                                   help="Total simulation duration. 0 = single snapshot.")
    time_step = st.number_input("Time Step (s)", min_value=1.0, max_value=60.0,
                                value=10.0, step=1.0,
                                help="Interval between frames.")

    st.divider()
    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# Tabs layout
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════
# Time frame selector (visible after simulation)
# ════════════════════════════════════════════════════════════════════

if "sim_results" in st.session_state and len(st.session_state["sim_results"]["timesteps"]) > 1:
    ts_list = st.session_state["sim_results"]["timesteps"]
    selected_t = st.select_slider(
        "⏱️ Time",
        options=ts_list,
        format_func=lambda t: f"t = {t:.1f} s",
        key="time_slider",
    )
    selected_frame_idx = ts_list.index(selected_t)
else:
    selected_frame_idx = 0

tab_scenario, tab_processing, tab_fusion, tab_intercept, tab_recommend = st.tabs([
    "🗺️ Scenario",
    "📊 Signal Processing",
    "🔗 Fusion",
    "🎯 Intercept Assessment",
    "📋 Recommendation",
])

# ════════════════════════════════════════════════════════════════════
# Build objects from sidebar config (no computation yet)
# ════════════════════════════════════════════════════════════════════

def build_objects():
    radars = [
        Radar(
            position=[c["x"], c["y"], c["z"]],
            fc=c["fc"],
            tx_power_dBm=c["power"],
            antenna_gain_dB=c["gain"],
            antenna_beamwidth=c["bw"],
            look_azimuth=c["az"],
            look_elevation=c["el"],
        )
        for c in radar_cfgs
    ]
    targets = [
        Target(
            position=[c["x"], c["y"], c["z"]],
            velocity=[c["vx"], c["vy"], c["vz"]],
            rcs_dbsm=c["rcs"],
        )
        for c in target_cfgs
    ]
    systems = [
        InterceptorSystem(
            name=c["name"],
            position=[c["x"], c["y"], c["z"]],
            max_range=c["maxr"],
            min_range=c["minr"],
            max_target_velocity=c["maxv"],
            reaction_time=c["rt"],
            salvo_size=c["sv"],
        )
        for c in system_cfgs
    ]
    return radars, targets, systems


# ════════════════════════════════════════════════════════════════════
# Scenario tab — live preview at t=0; after sim show selected frame + trails
# ════════════════════════════════════════════════════════════════════

with tab_scenario:
    radars, targets, systems = build_objects()

    # If simulation has run, show the selected frame with trajectory trails
    if "sim_results" in st.session_state:
        sim = st.session_state["sim_results"]
        frame = sim["frames"][selected_frame_idx]
        frame_targets = frame["targets"]
        frame_t = frame["t"]
        orig_targets = sim["original_targets"]

        scenario = Scenario(sim["radars"], frame_targets)
        geometry = frame["geometry"]

        st.subheader(f"Scenario Overview — t = {frame_t:.1f} s")
        fig, ax = scenario.plot_scenario(geometry, interceptor_systems=systems,
                                         save_path=None)
        # Draw trajectory trails
        for i, orig in enumerate(orig_targets):
            trail_x = [orig.position[0] + orig.velocity[0] * ft["t"]
                        for ft in sim["frames"]]
            trail_y = [orig.position[1] + orig.velocity[1] * ft["t"]
                        for ft in sim["frames"]]
            ax.plot(trail_x, trail_y, ":", color="blue", alpha=0.35, linewidth=1.2)
            # start marker
            ax.plot(trail_x[0], trail_y[0], "x", color="gray", markersize=5, alpha=0.5)

        ax.set_title(f"Scenario — t = {frame_t:.1f} s")
        apply_consistent_limits(ax, sim["radars"], frame_targets, systems)
        draw_border_line(ax, sim["radars"], frame_targets, systems)
        fig_to_st(fig)
    else:
        # Live preview at t=0 (no simulation yet)
        scenario = Scenario(radars, targets)
        geometry = scenario.compute_geometry()

        st.subheader("Scenario Overview (t = 0 — preview)")
        fig, ax = scenario.plot_scenario(geometry, interceptor_systems=systems,
                                         save_path=None)
        apply_consistent_limits(ax, radars, targets, systems)
        draw_border_line(ax, radars, targets, systems)
        fig_to_st(fig)

    # Geometry table
    st.subheader("Radar–Target Geometry")
    import pandas as pd
    if "sim_results" in st.session_state:
        geometry = st.session_state["sim_results"]["frames"][selected_frame_idx]["geometry"]
    rows = []
    for r_idx, geom_list in geometry.items():
        for g in geom_list:
            rows.append({
                "Radar": r_idx,
                "Target": g.target_index,
                "Range (m)": round(g.range_m, 1),
                "Radial Vel (m/s)": round(g.radial_velocity, 2),
                "Azimuth (°)": round(g.azimuth_deg, 2),
                "Elevation (°)": round(g.elevation_deg, 2),
                "Antenna Gain (dB)": round(g.antenna_gain_dB, 2),
                "RCS (dBsm)": round(g.rcs_dbsm, 1),
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# Run simulation on button press → store in session_state
# ════════════════════════════════════════════════════════════════════

if run_btn:
    radars, targets, systems = build_objects()

    # ── Build timestep list ──────────────────────────────────────────
    if sim_duration <= 0:
        timesteps = [0.0]
    else:
        timesteps = list(np.arange(0.0, sim_duration + time_step * 0.5, time_step))

    n_frames = len(timesteps)

    # Waveform / noise / processing templates (one per radar, cycling defaults)
    waveform_templates = [
        dict(bw=10e6, pd=10e-6, prf=5000, np_=256, ns=1024),
        dict(bw=8e6,  pd=12e-6, prf=4000, np_=128, ns=512),
        dict(bw=12e6, pd=8e-6,  prf=6000, np_=256, ns=1024),
    ]
    processing_templates = [
        dict(rw=WindowType.HAMMING, dw=WindowType.HANNING, gc=4, tc=16, tf=20.0, mo=1),
        dict(rw=WindowType.HANNING, dw=WindowType.HAMMING, gc=3, tc=12, tf=18.0, mo=1),
        dict(rw=WindowType.HAMMING, dw=WindowType.HANNING, gc=4, tc=16, tf=22.0, mo=1),
    ]

    ml_available = use_ml and os.path.exists("xgb_intercept_model.json")
    if use_ml and not ml_available:
        st.warning("XGBoost model not found. Showing analytical results only.")

    frames = []
    total_steps = n_frames * len(radars)
    progress = st.progress(0)
    status_text = st.empty()

    for f_idx, t in enumerate(timesteps):
        status_text.text(f"Frame {f_idx + 1}/{n_frames}  —  t = {t:.1f} s")

        # Advance target positions by velocity * t
        frame_targets = [
            Target(
                position=tgt.position + tgt.velocity * t,
                velocity=tgt.velocity,
                rcs_dbsm=tgt.rcs_dbsm,
            )
            for tgt in targets
        ]

        scenario = Scenario(radars, frame_targets)
        geometry = scenario.compute_geometry()

        per_radar_results = {}
        per_radar_detections = {}

        for r_idx, radar in enumerate(radars):
            wt = waveform_templates[r_idx % len(waveform_templates)]
            pt = processing_templates[r_idx % len(processing_templates)]

            waveform = WaveformConfig(
                waveform_type="lfm",
                bandwidth=wt["bw"],
                pulse_duration=wt["pd"],
                PRF=wt["prf"],
                N_pulses=wt["np_"],
                N_samples=wt["ns"],
            )
            noise = NoiseConfig(
                thermal_noise_power=0.01,
                clutter_enabled=use_clutter,
                clutter_cnr_dB=15,
                clutter_correlation=0.98,
                clutter_profile="range_dependent",
            )
            proc_cfg = ProcessingConfig(
                range_window=pt["rw"],
                doppler_window=pt["dw"],
                cfar_guard_cells=pt["gc"],
                cfar_training_cells=pt["tc"],
                cfar_threshold_factor=pt["tf"],
                mti_enabled=use_mti,
                mti_order=pt["mo"],
            )

            sig_gen = SignalGenerator(radar, waveform, noise, geometry[r_idx])
            rx_signal, tx_ref = sig_gen.get_signal()

            processor = RadarProcessor(waveform, proc_cfg)
            results = processor.process(rx_signal, tx_ref, radar.wavelength)

            per_radar_results[r_idx] = {
                "results": results,
                "waveform": waveform,
            }
            per_radar_detections[r_idx] = results["estimated_targets"]

            done = f_idx * len(radars) + r_idx + 1
            progress.progress(done / total_steps)

        fused_targets = associate_and_fuse(radars, per_radar_detections)

        blackbox_ana = InterceptBlackbox(systems, use_ml=False)
        P_analytical = blackbox_ana.evaluate(fused_targets)

        P_ml = None
        if ml_available:
            blackbox_ml = InterceptBlackbox(systems, use_ml=True)
            P_ml = blackbox_ml.evaluate(fused_targets)

        frames.append({
            "t": t,
            "targets": frame_targets,
            "geometry": geometry,
            "per_radar_results": per_radar_results,
            "fused_targets": fused_targets,
            "P_analytical": P_analytical,
            "P_ml": P_ml,
        })

    status_text.empty()
    progress.empty()

    st.session_state["sim_results"] = {
        "radars": radars,
        "original_targets": targets,
        "systems": systems,
        "timesteps": timesteps,
        "frames": frames,
    }
    st.rerun()

# ════════════════════════════════════════════════════════════════════
# Processing tab
# ════════════════════════════════════════════════════════════════════

with tab_processing:
    if "sim_results" not in st.session_state:
        st.info("Press **▶ Run Simulation** in the sidebar to see results.")
    else:
        import pandas as pd
        sim = st.session_state["sim_results"]
        frame = sim["frames"][selected_frame_idx]
        st.caption(f"Showing frame at t = {frame['t']:.1f} s")
        for r_idx, data in frame["per_radar_results"].items():
            r = data["results"]
            estimated = r["estimated_targets"]

            st.subheader(f"Radar {r_idx} — {sim['radars'][r_idx].fc/1e9:.1f} GHz")
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = plot_range_doppler(r, r_idx)
                fig_to_st(fig)
            with col2:
                st.markdown("**Detections**")
                if estimated:
                    det_rows = [
                        {
                            "Range (m)": round(d["range"], 1),
                            "Velocity (m/s)": round(d["velocity"], 2),
                            "Power (dB)": round(d["power_dB"], 1),
                        }
                        for d in estimated
                    ]
                    st.dataframe(pd.DataFrame(det_rows), use_container_width=True)
                else:
                    st.write("No detections.")
            st.divider()

# ════════════════════════════════════════════════════════════════════
# Fusion tab
# ════════════════════════════════════════════════════════════════════

with tab_fusion:
    if "sim_results" not in st.session_state:
        st.info("Press **▶ Run Simulation** in the sidebar to see results.")
    else:
        import pandas as pd
        sim = st.session_state["sim_results"]
        frame = sim["frames"][selected_frame_idx]
        fused = frame["fused_targets"]

        st.subheader(f"Fused Targets ({len(fused)} detected) — t = {frame['t']:.1f} s")

        if not fused:
            st.warning("No fused targets detected.")
        else:
            rows = []
            for ft in fused:
                rows.append({
                    "ID": f"FT{ft.fused_index}",
                    "X (m)": round(ft.position[0], 1),
                    "Y (m)": round(ft.position[1], 1),
                    "Z (m)": round(ft.position[2], 1),
                    "Vx (m/s)": round(ft.velocity_vector[0], 2),
                    "Vy (m/s)": round(ft.velocity_vector[1], 2),
                    "Vz (m/s)": round(ft.velocity_vector[2], 2),
                    "Speed (m/s)": round(float(np.linalg.norm(ft.velocity_vector)), 2),
                    "Radars": ft.n_radars,
                    "Track Quality": round(ft.track_quality, 3),
                    "Position Method": ft.position_method,
                    "Power (dB)": round(ft.power_dB, 1),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Feature vector inspection
            st.subheader("Feature Vectors (per system–target pair)")
            systems = sim["systems"]
            feat_rows = []
            for ft in fused:
                for sys in systems:
                    fv = extract_from_system_target(sys, ft)
                    row = {"System": sys.name, "Target": f"FT{ft.fused_index}"}
                    row.update(dict(zip(FEATURE_NAMES, [round(float(v), 4) for v in fv])))
                    feat_rows.append(row)
            st.dataframe(pd.DataFrame(feat_rows), use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# Intercept Assessment tab
# ════════════════════════════════════════════════════════════════════

with tab_intercept:
    if "sim_results" not in st.session_state:
        st.info("Press **▶ Run Simulation** in the sidebar to see results.")
    else:
        import pandas as pd
        sim = st.session_state["sim_results"]
        frame = sim["frames"][selected_frame_idx]
        systems = sim["systems"]
        fused = frame["fused_targets"]
        P_ana = frame["P_analytical"]
        P_ml = frame["P_ml"]

        if not fused:
            st.warning("No fused targets to assess.")
        else:
            st.caption(f"Showing frame at t = {frame['t']:.1f} s")

            # ── Analytical ──────────────────────────────────────────
            st.subheader("Analytical Model")
            ana_fig = plot_intercept_heatmap(systems, fused, P_ana,
                                             f"Analytical Intercept Probability (t={frame['t']:.1f}s)")
            fig_to_st(ana_fig)

            # ── ML ──────────────────────────────────────────────────
            if P_ml is not None:
                st.subheader("XGBoost ML Model")
                ml_fig = plot_intercept_heatmap(systems, fused, P_ml,
                                                f"ML Intercept Probability (t={frame['t']:.1f}s)")
                fig_to_st(ml_fig)

                # Comparison scatter
                st.subheader("Analytical vs ML Comparison")
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    scatter_fig = plot_comparison_scatter(P_ana, P_ml)
                    fig_to_st(scatter_fig)
                with col_b:
                    diff = np.abs(P_ana - P_ml) * 100
                    corr = np.corrcoef(P_ana.ravel(), P_ml.ravel())[0, 1] if P_ana.size > 1 else float("nan")
                    st.metric("Mean Absolute Difference", f"{diff.mean():.1f}%")
                    st.metric("Max Absolute Difference", f"{diff.max():.1f}%")
                    st.metric("Correlation", f"{corr:.4f}")

            # ── Probability table ────────────────────────────────────
            st.subheader("Full Probability Table")
            prob_rows = []
            for i, sys_obj in enumerate(systems):
                row = {"System": sys_obj.name}
                for j, ft in enumerate(fused):
                    row[f"FT{ft.fused_index} Ana"] = f"{P_ana[i, j]*100:.1f}%"
                    if P_ml is not None:
                        row[f"FT{ft.fused_index} ML"] = f"{P_ml[i, j]*100:.1f}%"
                prob_rows.append(row)
            st.dataframe(pd.DataFrame(prob_rows), use_container_width=True)

            # ── P(intercept) over time chart ─────────────────────────
            if len(sim["timesteps"]) > 1:
                st.subheader("Intercept Probability Over Time")
                ts = sim["timesteps"]

                # Build per-system mean analytical P over time
                fig_time, ax_time = plt.subplots(figsize=(10, 4.5))
                for s_idx, sys_obj in enumerate(systems):
                    p_over_time = []
                    for f in sim["frames"]:
                        p_mat = f["P_analytical"]
                        if p_mat.size > 0 and s_idx < p_mat.shape[0]:
                            p_over_time.append(np.mean(p_mat[s_idx, :]) * 100)
                        else:
                            p_over_time.append(0.0)
                    ax_time.plot(ts, p_over_time, "o-", label=sys_obj.name, markersize=4)

                ax_time.set_xlabel("Time (s)")
                ax_time.set_ylabel("Mean P(intercept) [%]")
                ax_time.set_title("Analytical — Mean Intercept Probability per System Over Time")
                ax_time.legend(fontsize=7, loc="best")
                ax_time.grid(True, alpha=0.3)
                ax_time.set_xlim(ts[0], ts[-1])
                ax_time.set_ylim(-2, 102)
                plt.tight_layout()
                fig_to_st(fig_time)

# ════════════════════════════════════════════════════════════════════
# Recommendation tab
# ════════════════════════════════════════════════════════════════════

with tab_recommend:
    if "sim_results" not in st.session_state:
        st.info("Press **▶ Run Simulation** in the sidebar to see results.")
    else:
        import pandas as pd
        sim = st.session_state["sim_results"]
        frame = sim["frames"][selected_frame_idx]
        systems = sim["systems"]
        radars_sim = sim["radars"]
        fused = frame["fused_targets"]
        P_ml = frame["P_ml"]
        P_ana = frame["P_analytical"]

        # Use ML if available, else analytical
        P = P_ml if P_ml is not None else P_ana
        prob_label = "ML" if P_ml is not None else "Analytical"

        if not fused:
            st.warning("No fused targets detected — cannot generate recommendations.")
        else:
            st.subheader(f"Best Interceptor Recommendation — t = {frame['t']:.1f} s")
            st.caption(f"Probability source: **{prob_label}**")

            # ── Build recommendation map ─────────────────────────────
            fig, ax = plt.subplots(figsize=(13, 9))

            # Draw radars
            radar_colors = plt.cm.Set1(np.linspace(0, 1, max(len(radars_sim), 3)))
            for r_idx, radar in enumerate(radars_sim):
                rx, ry = radar.position[0], radar.position[1]
                ax.plot(rx, ry, "s", color=radar_colors[r_idx], markersize=11,
                        label=f"Radar {r_idx} ({radar.fc/1e9:.1f} GHz)")
                ax.annotate(f"R{r_idx}", (rx, ry),
                            textcoords="offset points", xytext=(-12, -12),
                            fontsize=7, color=radar_colors[r_idx], fontweight="bold")

            # Draw interceptor systems
            intc_colors = plt.cm.Dark2(np.linspace(0, 1, max(len(systems), 3)))
            for s_idx, sys_obj in enumerate(systems):
                sx, sy = sys_obj.position[0], sys_obj.position[1]
                ax.plot(sx, sy, "^", color=intc_colors[s_idx], markersize=12,
                        label=sys_obj.name)
                ax.annotate(sys_obj.name, (sx, sy),
                            textcoords="offset points", xytext=(8, -14),
                            fontsize=6, color=intc_colors[s_idx], fontweight="bold")

            # Draw fused targets and best interceptor connections
            recommendations = []
            for j, ft in enumerate(fused):
                tx, ty = ft.position[0], ft.position[1]
                ax.plot(tx, ty, "o", color="crimson", markersize=10, zorder=5)
                ax.annotate(f"FT{ft.fused_index}",
                            (tx, ty), textcoords="offset points", xytext=(10, 6),
                            fontsize=8, color="crimson", fontweight="bold")

                # Find best interceptor for this target
                if j < P.shape[1]:
                    probs_for_target = P[:, j]
                    best_idx = int(np.argmax(probs_for_target))
                    best_prob = probs_for_target[best_idx]

                    if best_prob > 0:
                        best_sys = systems[best_idx]
                        sx, sy = best_sys.position[0], best_sys.position[1]

                        # Red line from target to best interceptor
                        ax.plot([tx, sx], [ty, sy], '-', color='red',
                                linewidth=2.0, alpha=0.7, zorder=3)

                        # Probability label at midpoint
                        mid_x, mid_y = (tx + sx) / 2, (ty + sy) / 2
                        ax.text(mid_x, mid_y, f"{best_prob*100:.1f}%",
                                ha="center", va="center", fontsize=8,
                                fontweight="bold", color="white",
                                bbox=dict(boxstyle="round,pad=0.3",
                                          facecolor="red", alpha=0.8))

                        recommendations.append({
                            "Target": f"FT{ft.fused_index}",
                            "Position": f"({ft.position[0]:.0f}, {ft.position[1]:.0f})",
                            "Best Interceptor": best_sys.name,
                            "P(intercept)": f"{best_prob*100:.1f}%",
                        })
                    else:
                        # No viable interceptor
                        ax.annotate("No viable\ninterceptor",
                                    (tx, ty), textcoords="offset points",
                                    xytext=(10, -18), fontsize=6,
                                    color="gray", fontstyle="italic")
                        recommendations.append({
                            "Target": f"FT{ft.fused_index}",
                            "Position": f"({ft.position[0]:.0f}, {ft.position[1]:.0f})",
                            "Best Interceptor": "—",
                            "P(intercept)": "0.0%",
                        })

            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title("Tactical Recommendation Map — Best Interceptor per Target")
            ax.set_aspect("equal")
            ax.legend(loc="upper left", fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_to_st(fig)

            # ── Summary table ────────────────────────────────────────
            st.subheader("Recommendation Summary")
            if recommendations:
                st.dataframe(pd.DataFrame(recommendations), use_container_width=True)
