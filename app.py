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
    n_radars = st.number_input("Number of radars", min_value=1, max_value=5,
                               value=3, step=1)

    radar_defaults = [
        dict(x=0,     y=0,    z=0, fc=10.0, power=60, gain=30, bw=5.0,  az=10.0,  el=0.0),
        dict(x=0,     y=3000, z=0, fc=9.0,  power=58, gain=30, bw=8.0,  az=-25.0, el=0.0),
        dict(x=-2000, y=1500, z=0, fc=10.0, power=60, gain=30, bw=8.0,  az=0.0,   el=0.0),
        dict(x=1000,  y=5000, z=0, fc=9.5,  power=55, gain=28, bw=6.0,  az=45.0,  el=0.0),
        dict(x=3000,  y=2000, z=0, fc=10.5, power=62, gain=32, bw=4.0,  az=-10.0, el=0.0),
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
    n_targets = st.number_input("Number of targets", min_value=1, max_value=8,
                                value=5, step=1)

    target_defaults = [
        dict(x=5000,  y=800,  z=0,  vx=25,  vy=5,   vz=0, rcs=10),
        dict(x=3000,  y=200,  z=50, vx=-15, vy=0,   vz=0, rcs=5),
        dict(x=7000,  y=1500, z=0,  vx=20,  vy=-10, vz=0, rcs=15),
        dict(x=2000,  y=100,  z=0,  vx=10,  vy=2,   vz=0, rcs=0),
        dict(x=6000,  y=-500, z=0,  vx=-30, vy=8,   vz=0, rcs=8),
        dict(x=4000,  y=3000, z=100,vx=15,  vy=-5,  vz=0, rcs=5),
        dict(x=8000,  y=500,  z=0,  vx=-20, vy=10,  vz=0, rcs=12),
        dict(x=1500,  y=2500, z=50, vx=30,  vy=0,   vz=0, rcs=3),
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
    n_systems = st.number_input("Number of systems", min_value=1, max_value=6,
                                value=3, step=1)

    sys_defaults = [
        dict(name="SAM Alpha",   x=1000,  y=1000, z=0, maxr=8000,  minr=500,  maxv=400, rt=5.0, sv=2),
        dict(name="SAM Beta",    x=-1000, y=2500, z=0, maxr=12000, minr=300,  maxv=600, rt=3.0, sv=3),
        dict(name="CIWS Delta",  x=500,   y=500,  z=0, maxr=2000,  minr=100,  maxv=800, rt=1.0, sv=4),
        dict(name="SAM Gamma",   x=2000,  y=4000, z=0, maxr=10000, minr=400,  maxv=500, rt=4.0, sv=2),
        dict(name="SAM Epsilon", x=-500,  y=3500, z=0, maxr=15000, minr=600,  maxv=700, rt=6.0, sv=1),
        dict(name="CIWS Zeta",   x=1500,  y=-500, z=0, maxr=1500,  minr=100,  maxv=900, rt=0.5, sv=6),
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

    st.divider()
    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# Tabs layout
# ════════════════════════════════════════════════════════════════════

tab_scenario, tab_processing, tab_fusion, tab_intercept = st.tabs([
    "🗺️ Scenario",
    "📊 Signal Processing",
    "🔗 Fusion",
    "🎯 Intercept Assessment",
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
# Scenario tab — always live (no heavy compute)
# ════════════════════════════════════════════════════════════════════

with tab_scenario:
    radars, targets, systems = build_objects()
    scenario = Scenario(radars, targets)
    geometry = scenario.compute_geometry()

    st.subheader("Scenario Overview")
    fig = plot_scenario_figure(scenario, geometry, systems)
    fig_to_st(fig)

    # Geometry table
    st.subheader("Radar–Target Geometry")
    import pandas as pd
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
    scenario = Scenario(radars, targets)
    geometry = scenario.compute_geometry()

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

    per_radar_results = {}
    per_radar_detections = {}

    with st.spinner("Running signal generation & processing…"):
        progress = st.progress(0)
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
            progress.progress((r_idx + 1) / len(radars))

    with st.spinner("Running multi-radar fusion…"):
        fused_targets = associate_and_fuse(radars, per_radar_detections)

    with st.spinner("Computing intercept probabilities…"):
        blackbox_ana = InterceptBlackbox(systems, use_ml=False)
        P_analytical = blackbox_ana.evaluate(fused_targets)

        ml_available = use_ml and os.path.exists("xgb_intercept_model.json")
        if use_ml and not ml_available:
            st.warning("XGBoost model not found. Showing analytical results only.")
        P_ml = None
        if ml_available:
            blackbox_ml = InterceptBlackbox(systems, use_ml=True)
            P_ml = blackbox_ml.evaluate(fused_targets)

    st.session_state["sim_results"] = {
        "radars": radars,
        "targets": targets,
        "systems": systems,
        "geometry": geometry,
        "per_radar_results": per_radar_results,
        "fused_targets": fused_targets,
        "P_analytical": P_analytical,
        "P_ml": P_ml,
    }
    st.success(f"Simulation complete — {len(fused_targets)} fused target(s) detected.")

# ════════════════════════════════════════════════════════════════════
# Processing tab
# ════════════════════════════════════════════════════════════════════

with tab_processing:
    if "sim_results" not in st.session_state:
        st.info("Press **▶ Run Simulation** in the sidebar to see results.")
    else:
        import pandas as pd
        res = st.session_state["sim_results"]
        for r_idx, data in res["per_radar_results"].items():
            r = data["results"]
            estimated = r["estimated_targets"]

            st.subheader(f"Radar {r_idx} — {res['radars'][r_idx].fc/1e9:.1f} GHz")
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
        res = st.session_state["sim_results"]
        fused = res["fused_targets"]

        st.subheader(f"Fused Targets ({len(fused)} detected)")

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
            systems = res["systems"]
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
        res = st.session_state["sim_results"]
        systems = res["systems"]
        fused = res["fused_targets"]
        P_ana = res["P_analytical"]
        P_ml = res["P_ml"]

        if not fused:
            st.warning("No fused targets to assess.")
        else:
            # ── Analytical ──────────────────────────────────────────
            st.subheader("Analytical Model")
            ana_fig = plot_intercept_heatmap(systems, fused, P_ana,
                                             "Analytical Intercept Probability")
            fig_to_st(ana_fig)

            # ── ML ──────────────────────────────────────────────────
            if P_ml is not None:
                st.subheader("XGBoost ML Model")
                ml_fig = plot_intercept_heatmap(systems, fused, P_ml,
                                                "ML Intercept Probability")
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
            for i, sys in enumerate(systems):
                row = {"System": sys.name}
                for j, ft in enumerate(fused):
                    row[f"FT{ft.fused_index} Ana"] = f"{P_ana[i, j]*100:.1f}%"
                    if P_ml is not None:
                        row[f"FT{ft.fused_index} ML"] = f"{P_ml[i, j]*100:.1f}%"
                prob_rows.append(row)
            st.dataframe(pd.DataFrame(prob_rows), use_container_width=True)
