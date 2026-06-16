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
import plotly.graph_objects as go
import streamlit as st

# ── ensure workspace is on the path ─────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from scenario import Radar, Target, Scenario
from signal_gen import WaveformConfig, NoiseConfig, SignalGenerator
from processing import ProcessingConfig, WindowType, RadarProcessor
from fusion import associate_and_fuse, FusedTarget
from interceptor import InterceptorSystem, InterceptBlackbox
from feature_extraction import FEATURE_NAMES, extract_from_system_target
from classifier import extract_doppler_features, classify_target

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
# Plotly animated tactical chart builders
# ════════════════════════════════════════════════════════════════════

# Coordinate conversion: simulation metres -> geographic lat/lon
_MAP_CENTER_LAT = 25.1   # Dubai inland (coast ~10 km north)
_MAP_CENTER_LON = 55.35
_M_PER_DEG_LAT  = 111_000.0
_M_PER_DEG_LON  = _M_PER_DEG_LAT * np.cos(np.radians(_MAP_CENTER_LAT))

def _mlat(y_m): return _MAP_CENTER_LAT + float(y_m) / _M_PER_DEG_LAT
def _mlon(x_m): return _MAP_CENTER_LON + float(x_m) / _M_PER_DEG_LON

_TYPE_COLOR = {
    "drone":      "#ffb347",
    "helicopter": "#adff2f",
    "fixed_wing": "#ff4444",
}
_DEFAULT_TGT_COLOR = "#ff8888"

def _tgt_colors(tgt_list):
    return [_TYPE_COLOR.get(getattr(t, "target_type", None), _DEFAULT_TGT_COLOR)
            for t in tgt_list]


def _export_gif(fig: go.Figure, fps: int = 5, zoom_boost: float = 3.0):
    """Render each Plotly frame to PNG and combine into an animated GIF.

    zoom_boost: added to the mapbox zoom level so the GIF is zoomed in.
    Returns io.BytesIO on success, None if kaleido / Pillow is missing.
    """
    try:
        import copy
        import plotly.io as pio
        from PIL import Image as _PImage
    except ImportError:
        return None
    if not fig.frames:
        return None

    base_fig = go.Figure(data=fig.data, layout=fig.layout)
    base_fig.update_layout(updatemenus=[], sliders=[])
    base_dict = base_fig.to_dict()
    if "mapbox" in base_dict.get("layout", {}) and "zoom" in base_dict["layout"].get("mapbox", {}):
        base_dict["layout"]["mapbox"]["zoom"] += zoom_boost

    pngs = []
    for frame in fig.frames:
        d = copy.deepcopy(base_dict)
        trace_indices = list(frame.traces) if frame.traces else list(range(len(frame.data)))
        for trace_upd, idx in zip(frame.data, trace_indices):
            if idx < len(d["data"]):
                d["data"][idx].update(trace_upd.to_plotly_json())
        if frame.layout:
            d["layout"].update(frame.layout.to_plotly_json())
        tmp = go.Figure(d)
        pngs.append(pio.to_image(tmp, format="png", width=1200, height=720, scale=1))

    if not pngs:
        return None
    images = [_PImage.open(io.BytesIO(p)).convert("RGB") for p in pngs]
    buf = io.BytesIO()
    images[0].save(
        buf, format="GIF", save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0, optimize=False,
    )
    buf.seek(0)
    return buf


def build_scenario_plotly(sim_results: dict, systems: list):
    """Build an animated Plotly tactical map on a real geographic map background."""
    radars           = sim_results["radars"]
    original_targets = sim_results["original_targets"]
    frames_data      = sim_results["frames"]
    n_targets        = len(original_targets)

    PAPER    = "#070e1a"
    TEXT_COL = "#a8d4f5"
    CYAN     = "#00cfff"
    GREEN    = "#00e87c"
    DARK_RED = "#cc0000"

    # Spatial bounds
    all_x, all_y = [], []
    for r in radars:
        all_x.append(float(r.position[0])); all_y.append(float(r.position[1]))
    for s in systems:
        all_x.extend([float(s.position[0] + s.max_range),
                       float(s.position[0] - s.max_range)])
        all_y.extend([float(s.position[1] + s.max_range),
                       float(s.position[1] - s.max_range)])
    for fd in frames_data:
        for tgt in fd["targets"]:
            all_x.append(float(tgt.position[0])); all_y.append(float(tgt.position[1]))
    if not all_x: all_x = [0.0, 10000.0]
    if not all_y: all_y = [0.0, 10000.0]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_pad = max((x_max - x_min) * 0.15, 2000.0)
    y_pad = max((y_max - y_min) * 0.15, 2000.0)
    x_min -= x_pad; x_max += x_pad
    y_min -= y_pad; y_max += y_pad

    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    lon_span = _mlon(x_max) - _mlon(x_min)
    lat_span = _mlat(y_max) - _mlat(y_min)
    zoom = float(np.clip(np.log2(3.6 / max(lon_span, lat_span, 0.001)), 4, 13))

    # Beam-wedge scale from t=0 geometry
    first_geom = frames_data[0]["geometry"]
    all_ranges = [g.range_m for glist in first_geom.values() for g in glist]
    max_r = float(max(all_ranges)) * 1.2 if all_ranges else 10000.0

    theta = np.linspace(0, 2 * np.pi, 120)

    # ═══════════════════════ STATIC TRACES ═══════════════════════════
    traces = []

    # 1) Radar positions
    traces.append(go.Scattermapbox(
        lat=[_mlat(r.position[1]) for r in radars],
        lon=[_mlon(r.position[0]) for r in radars],
        mode="markers+text",
        marker=dict(size=16, color=CYAN, opacity=0.95),
        text=[f"◆ RDR-{i:02d}" for i in range(len(radars))],
        textposition="top right",
        textfont=dict(color=CYAN, size=10),
        hovertext=[
            f"<b>RADAR {i:02d}</b><br>"
            f"{r.fc/1e9:.2f} GHz  Pt={r.tx_power_dBm:.0f} dBm  G={r.antenna_gain_dB:.0f} dB<br>"
            f"BW={r.antenna_beamwidth:.1f}°  Az={r.look_azimuth:.1f}°"
            for i, r in enumerate(radars)
        ],
        hoverinfo="text", name="Radars", showlegend=True,
    ))

    # 2) Beam wedges (filled polygon, one per radar)
    for radar in radars:
        rx, ry = float(radar.position[0]), float(radar.position[1])
        look_az_rad = np.radians(radar.look_azimuth)
        bw_rad      = np.radians(radar.antenna_beamwidth)
        angles = np.linspace(look_az_rad - bw_rad / 2, look_az_rad + bw_rad / 2, 50)
        wx = [rx] + list(rx + max_r * np.cos(angles)) + [rx]
        wy = [ry] + list(ry + max_r * np.sin(angles)) + [ry]
        traces.append(go.Scattermapbox(
            lat=[_mlat(y) for y in wy],
            lon=[_mlon(x) for x in wx],
            mode="lines",
            fill="toself", fillcolor="rgba(0,207,255,0.07)",
            line=dict(color=CYAN, width=1.0),
            hoverinfo="skip", showlegend=False,
        ))

    # 3) Interceptor positions
    traces.append(go.Scattermapbox(
        lat=[_mlat(s.position[1]) for s in systems],
        lon=[_mlon(s.position[0]) for s in systems],
        mode="markers+text",
        marker=dict(size=20, color=GREEN, opacity=0.95),
        text=[f"▲ {s.name[:10]}" for s in systems],
        textposition="bottom right",
        textfont=dict(color=GREEN, size=10),
        hovertext=[
            f"<b>{s.name}</b><br>"
            f"Max Range: {s.max_range/1000:.1f} km  Min: {s.min_range:.0f} m<br>"
            f"Max Vel: {s.max_target_velocity:.0f} m/s  RT: {s.reaction_time:.1f} s  Salvo: {s.salvo_size}"
            for s in systems
        ],
        hoverinfo="text", name="Interceptors", showlegend=True,
    ))

    # 4) Range rings (max + min per system) + km label trace
    for s in systems:
        sx, sy = float(s.position[0]), float(s.position[1])
        traces.append(go.Scattermapbox(
            lat=[_mlat(sy + s.max_range * np.sin(a)) for a in theta]
                + [_mlat(sy + s.max_range * np.sin(theta[0]))],
            lon=[_mlon(sx + s.max_range * np.cos(a)) for a in theta]
                + [_mlon(sx + s.max_range * np.cos(theta[0]))],
            mode="lines", line=dict(color=GREEN, width=1.2),
            opacity=0.35, hoverinfo="skip", showlegend=False,
        ))
        traces.append(go.Scattermapbox(
            lat=[_mlat(sy + s.min_range * np.sin(a)) for a in theta]
                + [_mlat(sy + s.min_range * np.sin(theta[0]))],
            lon=[_mlon(sx + s.min_range * np.cos(a)) for a in theta]
                + [_mlon(sx + s.min_range * np.cos(theta[0]))],
            mode="lines", line=dict(color=GREEN, width=0.6),
            opacity=0.20, hoverinfo="skip", showlegend=False,
        ))
        # km label at top of max ring
        traces.append(go.Scattermapbox(
            lat=[_mlat(sy + s.max_range)], lon=[_mlon(sx)],
            mode="text", text=[f"{s.max_range/1000:.0f} km"],
            textfont=dict(color=GREEN, size=9),
            hoverinfo="skip", showlegend=False,
        ))

    # 5) Border line (defence / threat zone divider)
    defense_xs    = ([float(r.position[0]) for r in radars]
                     + [float(s.position[0]) for s in systems])
    target_xs_all = [float(tgt.position[0])
                     for fd in frames_data for tgt in fd["targets"]]
    if defense_xs and target_xs_all:
        bx_mid = (max(defense_xs) + min(target_xs_all)) / 2.0
        y_span = np.linspace(y_min, y_max, 300)
        amp  = (y_max - y_min) * 0.02
        freq = 3.0 * np.pi / max(y_max - y_min, 1.0)
        bx_curve = bx_mid + amp * np.sin(freq * (y_span - y_min))
        traces.append(go.Scattermapbox(
            lat=[_mlat(y) for y in y_span],
            lon=[_mlon(x) for x in bx_curve],
            mode="lines",
            line=dict(color=DARK_RED, width=2.5),
            opacity=0.70, hoverinfo="skip", name="Threat Line", showlegend=True,
        ))

    n_static = len(traces)

    # Animated trace indices
    glow_idx    = n_static
    target_idx  = n_static + 1
    trail_start = n_static + 2
    pred_start  = n_static + 2 + n_targets
    vel_start   = n_static + 2 + 2 * n_targets

    PRED_TIME = 60.0

    # Initial (t=0) animated traces
    init_tgts   = frames_data[0]["targets"]
    init_colors = _tgt_colors(init_tgts)

    # Glow
    traces.append(go.Scattermapbox(
        lat=[_mlat(t.position[1]) for t in init_tgts],
        lon=[_mlon(t.position[0]) for t in init_tgts],
        mode="markers",
        marker=dict(size=30, color=init_colors, opacity=0.10),
        hoverinfo="skip", showlegend=False,
    ))

    # Target markers
    traces.append(go.Scattermapbox(
        lat=[_mlat(t.position[1]) for t in init_tgts],
        lon=[_mlon(t.position[0]) for t in init_tgts],
        mode="markers+text",
        marker=dict(size=12, color=init_colors, opacity=0.95),
        text=[f"T{i:02d}" for i in range(len(init_tgts))],
        textposition="top right",
        textfont=dict(color=TEXT_COL, size=10),
        hovertext=[
            f"<b>TGT-{i:02d}</b>  [{getattr(t, 'target_type', '?').upper()}]<br>"
            f"Pos: ({t.position[0]:.0f}, {t.position[1]:.0f}) m  "
            f"Alt: {t.position[2]:.0f} m<br>"
            f"Speed: {float(np.linalg.norm(t.velocity[:2])):.1f} m/s  "
            f"RCS: {t.rcs_dbsm:.1f} dBsm"
            for i, t in enumerate(init_tgts)
        ],
        hoverinfo="text", name="Targets", showlegend=True,
    ))

    # Trail traces (one per target)
    for _ in range(n_targets):
        traces.append(go.Scattermapbox(
            lat=[], lon=[], mode="lines",
            line=dict(color="#ff8c00", width=1.5),
            opacity=0.45, hoverinfo="skip", showlegend=False,
        ))

    # Predicted path traces (gold, 60 s forward from current position)
    for orig in original_targets:
        p0x, p0y = float(orig.position[0]), float(orig.position[1])
        p1x = p0x + float(orig.velocity[0]) * PRED_TIME
        p1y = p0y + float(orig.velocity[1]) * PRED_TIME
        traces.append(go.Scattermapbox(
            lat=[_mlat(p0y), _mlat(p1y)],
            lon=[_mlon(p0x), _mlon(p1x)],
            mode="lines",
            line=dict(color="#ffcc00", width=1.2),
            opacity=0.38, hoverinfo="skip", showlegend=False,
        ))

    # Velocity direction line traces (short, bright orange)
    for tgt in init_tgts:
        vel_mag = float(np.linalg.norm(tgt.velocity[:2]))
        scale   = max_r * 0.05 / (vel_mag + 1e-6)
        p0x, p0y = float(tgt.position[0]), float(tgt.position[1])
        p1x = p0x + float(tgt.velocity[0]) * scale
        p1y = p0y + float(tgt.velocity[1]) * scale
        traces.append(go.Scattermapbox(
            lat=[_mlat(p0y), _mlat(p1y)],
            lon=[_mlon(p0x), _mlon(p1x)],
            mode="lines",
            line=dict(color="#ff8c00", width=2.5),
            opacity=0.90, hoverinfo="skip", showlegend=False,
        ))

    # Plotly frames
    plotly_frames = []
    for f_idx, fd in enumerate(frames_data):
        t_val      = fd["t"]
        frame_tgts = fd["targets"]
        colors     = _tgt_colors(frame_tgts)

        frame_data_list  = []
        frame_trace_idxs = []

        # Glow
        frame_data_list.append(go.Scattermapbox(
            lat=[_mlat(t.position[1]) for t in frame_tgts],
            lon=[_mlon(t.position[0]) for t in frame_tgts],
            marker=dict(color=colors, size=30, opacity=0.10),
        ))
        frame_trace_idxs.append(glow_idx)

        # Target markers
        frame_data_list.append(go.Scattermapbox(
            lat=[_mlat(t.position[1]) for t in frame_tgts],
            lon=[_mlon(t.position[0]) for t in frame_tgts],
            text=[f"T{i:02d}" for i in range(len(frame_tgts))],
            marker=dict(color=colors, size=12, opacity=0.95),
            hovertext=[
                f"<b>TGT-{i:02d}</b>  [{getattr(t, 'target_type', '?').upper()}]<br>"
                f"Pos: ({t.position[0]:.0f}, {t.position[1]:.0f}) m  "
                f"Alt: {t.position[2]:.0f} m<br>"
                f"Speed: {float(np.linalg.norm(t.velocity[:2])):.1f} m/s  "
                f"RCS: {t.rcs_dbsm:.1f} dBsm"
                for i, t in enumerate(frame_tgts)
            ],
        ))
        frame_trace_idxs.append(target_idx)

        # Trails (grow with each frame)
        for ti, orig in enumerate(original_targets):
            tx_t = [float(orig.position[0] + orig.velocity[0] * frames_data[j]["t"])
                    for j in range(f_idx + 1)]
            ty_t = [float(orig.position[1] + orig.velocity[1] * frames_data[j]["t"])
                    for j in range(f_idx + 1)]
            frame_data_list.append(go.Scattermapbox(
                lat=[_mlat(y) for y in ty_t],
                lon=[_mlon(x) for x in tx_t],
            ))
            frame_trace_idxs.append(trail_start + ti)

        # Predicted paths (from current position)
        for ti, tgt in enumerate(frame_tgts):
            p0x, p0y = float(tgt.position[0]), float(tgt.position[1])
            p1x = p0x + float(tgt.velocity[0]) * PRED_TIME
            p1y = p0y + float(tgt.velocity[1]) * PRED_TIME
            frame_data_list.append(go.Scattermapbox(
                lat=[_mlat(p0y), _mlat(p1y)],
                lon=[_mlon(p0x), _mlon(p1x)],
            ))
            frame_trace_idxs.append(pred_start + ti)

        # Velocity direction lines
        for ti, tgt in enumerate(frame_tgts):
            vel_mag = float(np.linalg.norm(tgt.velocity[:2]))
            scale   = max_r * 0.05 / (vel_mag + 1e-6)
            p0x, p0y = float(tgt.position[0]), float(tgt.position[1])
            p1x = p0x + float(tgt.velocity[0]) * scale
            p1y = p0y + float(tgt.velocity[1]) * scale
            frame_data_list.append(go.Scattermapbox(
                lat=[_mlat(p0y), _mlat(p1y)],
                lon=[_mlon(p0x), _mlon(p1x)],
            ))
            frame_trace_idxs.append(vel_start + ti)

        plotly_frames.append(go.Frame(
            data=frame_data_list,
            traces=frame_trace_idxs,
            layout=go.Layout(title_text=f"\U0001f3af TACTICAL DISPLAY — T+{t_val:.0f}s"),
            name=f"{t_val:.1f}",
        ))

    # ── Insert interpolated frames for smooth playback ────────────────
    _sim_frame_names = {pf.name for pf in plotly_frames}
    if len(plotly_frames) > 1:
        _N_INTERP  = 4
        _interp_out = []
        for _fi in range(len(plotly_frames) - 1):
            _interp_out.append(plotly_frames[_fi])
            _fd0, _fd1   = frames_data[_fi], frames_data[_fi + 1]
            _t0,  _t1    = _fd0["t"], _fd1["t"]
            _tgts0, _tgts1 = _fd0["targets"], _fd1["targets"]
            for _k in range(1, _N_INTERP + 1):
                _a    = _k / (_N_INTERP + 1)
                _t_m  = _t0 + _a * (_t1 - _t0)
                _lats = [_mlat((1 - _a) * float(_ta.position[1]) + _a * float(_tb.position[1]))
                         for _ta, _tb in zip(_tgts0, _tgts1)]
                _lons = [_mlon((1 - _a) * float(_ta.position[0]) + _a * float(_tb.position[0]))
                         for _ta, _tb in zip(_tgts0, _tgts1)]
                _cols = _tgt_colors(_tgts0)
                _fd_m, _fi_m = [], []
                # Glow
                _fd_m.append(go.Scattermapbox(
                    lat=_lats, lon=_lons,
                    marker=dict(color=_cols, size=30, opacity=0.10)))
                _fi_m.append(glow_idx)
                # Targets
                _fd_m.append(go.Scattermapbox(
                    lat=_lats, lon=_lons,
                    text=[f"T{_i:02d}" for _i in range(len(_tgts0))],
                    marker=dict(color=_cols, size=12, opacity=0.95)))
                _fi_m.append(target_idx)
                # Trails
                for _ti, _orig in enumerate(original_targets):
                    _tx = ([float(_orig.position[0] + _orig.velocity[0] * frames_data[_j]["t"])
                            for _j in range(_fi + 1)]
                           + [float(_orig.position[0] + _orig.velocity[0] * _t_m)])
                    _ty = ([float(_orig.position[1] + _orig.velocity[1] * frames_data[_j]["t"])
                            for _j in range(_fi + 1)]
                           + [float(_orig.position[1] + _orig.velocity[1] * _t_m)])
                    _fd_m.append(go.Scattermapbox(
                        lat=[_mlat(_y) for _y in _ty],
                        lon=[_mlon(_x) for _x in _tx]))
                    _fi_m.append(trail_start + _ti)
                # Predicted paths
                for _ti, (_ta, _tb) in enumerate(zip(_tgts0, _tgts1)):
                    _p0x = (1 - _a) * float(_ta.position[0]) + _a * float(_tb.position[0])
                    _p0y = (1 - _a) * float(_ta.position[1]) + _a * float(_tb.position[1])
                    _fd_m.append(go.Scattermapbox(
                        lat=[_mlat(_p0y),
                             _mlat(_p0y + float(_ta.velocity[1]) * PRED_TIME)],
                        lon=[_mlon(_p0x),
                             _mlon(_p0x + float(_ta.velocity[0]) * PRED_TIME)]))
                    _fi_m.append(pred_start + _ti)
                # Velocity lines
                for _ti, (_ta, _tb) in enumerate(zip(_tgts0, _tgts1)):
                    _vm  = float(np.linalg.norm(_ta.velocity[:2]))
                    _sc  = max_r * 0.05 / (_vm + 1e-6)
                    _p0x = (1 - _a) * float(_ta.position[0]) + _a * float(_tb.position[0])
                    _p0y = (1 - _a) * float(_ta.position[1]) + _a * float(_tb.position[1])
                    _fd_m.append(go.Scattermapbox(
                        lat=[_mlat(_p0y), _mlat(_p0y + float(_ta.velocity[1]) * _sc)],
                        lon=[_mlon(_p0x), _mlon(_p0x + float(_ta.velocity[0]) * _sc)]))
                    _fi_m.append(vel_start + _ti)
                _interp_out.append(go.Frame(
                    data=_fd_m, traces=_fi_m,
                    layout=go.Layout(
                        title_text=f"\U0001f3af TACTICAL DISPLAY — T+{_t_m:.0f}s"),
                    name=f"{_t_m:.2f}",
                ))
        _interp_out.append(plotly_frames[-1])
        plotly_frames = _interp_out

    # Play / Pause / Slider
    has_anim    = len(plotly_frames) > 1
    updatemenus = []
    sliders_cfg = []

    if has_anim:
        updatemenus = [dict(
            type="buttons", showactive=False,
            x=0.05, xanchor="right", y=1.10, yanchor="top",
            bgcolor="#0f2340", bordercolor="#2a5080",
            font=dict(color=TEXT_COL, family="monospace"),
            buttons=[
                dict(label="▶  PLAY", method="animate",
                     args=[None, {"frame": {"duration": 150, "redraw": True},
                                  "fromcurrent": True,
                                  "transition": {"duration": 100,
                                                 "easing": "cubic-in-out"}}]),
                dict(label="⏸  PAUSE", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}]),
            ],
        )]
        sliders_cfg = [dict(
            active=0,
            currentvalue=dict(prefix="T+ ", suffix=" s",
                              font=dict(color=TEXT_COL, size=13, family="monospace"),
                              visible=True, xanchor="right"),
            pad=dict(b=10, t=55), len=0.85,
            x=0.1, y=0, xanchor="left", yanchor="top",
            bgcolor="#0f2340", bordercolor="#2a5080",
            font=dict(color=TEXT_COL, size=10, family="monospace"),
            steps=[
                dict(label=f"{fd['t']:.0f}", method="animate",
                     args=[[f"{fd['t']:.1f}"],
                           {"frame": {"duration": 100, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 80}}])
                for fd in frames_data
            ],
        )]

    layout = go.Layout(
        title=dict(
            text="\U0001f3af TACTICAL DISPLAY — T+0s",
            font=dict(color=TEXT_COL, size=15, family="monospace"),
            x=0.5, xanchor="center",
        ),
        paper_bgcolor=PAPER,
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=_mlat(cy), lon=_mlon(cx)),
            zoom=zoom,
        ),
        legend=dict(
            bgcolor="rgba(7,14,26,0.88)", bordercolor="#2a5080",
            borderwidth=1, font=dict(color=TEXT_COL, size=10, family="monospace"),
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ),
        updatemenus=updatemenus,
        sliders=sliders_cfg,
        margin=dict(l=0, r=0, t=60, b=60),
        height=700,
    )
    return go.Figure(data=traces, layout=layout, frames=plotly_frames)


def build_recommendation_plotly(sim_results: dict, systems: list):
    """Build an animated Plotly tactical recommendation map on a real geographic map."""
    radars      = sim_results["radars"]
    frames_data = sim_results["frames"]

    PAPER    = "#070e1a"
    TEXT_COL = "#a8d4f5"
    CYAN     = "#00cfff"
    GREEN    = "#00e87c"

    # Spatial bounds (fused targets + radars + interceptors)
    all_x, all_y = [], []
    for r in radars:
        all_x.append(float(r.position[0])); all_y.append(float(r.position[1]))
    for s in systems:
        all_x.append(float(s.position[0])); all_y.append(float(s.position[1]))
    for fd in frames_data:
        for ft in fd["fused_targets"]:
            all_x.append(float(ft.position[0])); all_y.append(float(ft.position[1]))
    if not all_x: all_x = [0.0, 10000.0]
    if not all_y: all_y = [0.0, 10000.0]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_pad = max((x_max - x_min) * 0.15, 2000.0)
    y_pad = max((y_max - y_min) * 0.15, 2000.0)
    x_min -= x_pad; x_max += x_pad
    y_min -= y_pad; y_max += y_pad

    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    lon_span = _mlon(x_max) - _mlon(x_min)
    lat_span = _mlat(y_max) - _mlat(y_min)
    zoom = float(np.clip(np.log2(3.6 / max(lon_span, lat_span, 0.001)), 4, 13))

    theta = np.linspace(0, 2 * np.pi, 120)

    # ═══════════════════════ STATIC TRACES ═══════════════════════════
    traces = []

    # Radars
    traces.append(go.Scattermapbox(
        lat=[_mlat(r.position[1]) for r in radars],
        lon=[_mlon(r.position[0]) for r in radars],
        mode="markers+text",
        marker=dict(size=14, color=CYAN, opacity=0.90),
        text=[f"◆ R{i:02d}" for i in range(len(radars))],
        textposition="top right",
        textfont=dict(color=CYAN, size=9),
        hovertext=[f"<b>RADAR {i:02d}</b><br>{r.fc/1e9:.2f} GHz"
                   for i, r in enumerate(radars)],
        hoverinfo="text", name="Radars", showlegend=True,
    ))

    # Interceptors
    traces.append(go.Scattermapbox(
        lat=[_mlat(s.position[1]) for s in systems],
        lon=[_mlon(s.position[0]) for s in systems],
        mode="markers+text",
        marker=dict(size=20, color=GREEN, opacity=0.95),
        text=[f"▲ {s.name[:10]}" for s in systems],
        textposition="bottom right",
        textfont=dict(color=GREEN, size=10),
        hovertext=[
            f"<b>{s.name}</b><br>"
            f"Max Range: {s.max_range/1000:.1f} km  Min: {s.min_range:.0f} m"
            for s in systems
        ],
        hoverinfo="text", name="Interceptors", showlegend=True,
    ))

    # Range rings (max, subtle)
    for s in systems:
        sx, sy = float(s.position[0]), float(s.position[1])
        traces.append(go.Scattermapbox(
            lat=[_mlat(sy + s.max_range * np.sin(a)) for a in theta]
                + [_mlat(sy + s.max_range * np.sin(theta[0]))],
            lon=[_mlon(sx + s.max_range * np.cos(a)) for a in theta]
                + [_mlon(sx + s.max_range * np.cos(theta[0]))],
            mode="lines", line=dict(color=GREEN, width=0.8),
            opacity=0.18, hoverinfo="skip", showlegend=False,
        ))

    n_static       = len(traces)
    glow_idx       = n_static
    fused_idx      = n_static + 1
    eng_hi_idx     = n_static + 2   # P >= 0.70  green
    eng_mid_idx    = n_static + 3   # 0.40 <= P < 0.70  yellow
    eng_lo_idx     = n_static + 4   # P < 0.40  red
    prob_label_idx = n_static + 5   # probability text labels

    # ── Helpers ──────────────────────────────────────────────────────
    def _prob_color_bg(p):
        if p >= 0.70: return "rgba(0,180,80,0.88)"
        if p >= 0.40: return "rgba(200,160,0,0.88)"
        return "rgba(200,20,20,0.88)"

    def _prob_labels(fused, P):
        """Return (lats, lons, texts) for probability label text trace."""
        lats, lons, texts = [], [], []
        if P is None or len(fused) == 0 or P.size == 0:
            return lats, lons, texts
        for j, ft in enumerate(fused):
            if j >= P.shape[1]:
                break
            best_i = int(np.argmax(P[:, j]))
            bp = float(P[best_i, j])
            if bp > 0.01:
                sx = float(systems[best_i].position[0])
                sy = float(systems[best_i].position[1])
                lats.append((_mlat(ft.position[1]) + _mlat(sy)) / 2)
                lons.append((_mlon(ft.position[0]) + _mlon(sx)) / 2)
                texts.append(f"{bp * 100:.0f}%")
        return lats, lons, texts

    def _eng_segments(fused, P, lo, hi):
        """Return (lat_list, lon_list) for engagements with lo <= P < hi."""
        lats, lons = [], []
        if P is None or len(fused) == 0 or P.size == 0:
            return lats, lons
        for j, ft in enumerate(fused):
            if j >= P.shape[1]:
                break
            best_i = int(np.argmax(P[:, j]))
            bp = float(P[best_i, j])
            if lo <= bp < hi:
                sx = float(systems[best_i].position[0])
                sy = float(systems[best_i].position[1])
                lats += [_mlat(ft.position[1]), _mlat(sy), None]
                lons += [_mlon(ft.position[0]), _mlon(sx), None]
        return lats, lons

    def _glow_colors(fused, P):
        colors = []
        if P is None or len(fused) == 0 or P.size == 0:
            return ["rgba(255,50,50,0.14)"] * len(fused)
        for j in range(len(fused)):
            if j >= P.shape[1]:
                colors.append("rgba(255,50,50,0.14)")
                continue
            mp = float(np.max(P[:, j]))
            if mp >= 0.70:
                colors.append("rgba(0,232,124,0.15)")
            elif mp >= 0.40:
                colors.append("rgba(255,215,0,0.15)")
            else:
                colors.append("rgba(255,50,50,0.16)")
        return colors

    def _ft_colors(fused):
        return [_TYPE_COLOR.get(ft.target_type, _DEFAULT_TGT_COLOR) for ft in fused]

    # Initial frame data
    init_fd    = frames_data[0]
    init_fused = init_fd["fused_targets"]
    init_P     = init_fd["P_ml"] if init_fd["P_ml"] is not None else init_fd["P_analytical"]

    gc = _glow_colors(init_fused, init_P)
    lh_lat, lh_lon = _eng_segments(init_fused, init_P, 0.70, 1.01)
    lm_lat, lm_lon = _eng_segments(init_fused, init_P, 0.40, 0.70)
    ll_lat, ll_lon = _eng_segments(init_fused, init_P, 0.00, 0.40)
    pl_lat, pl_lon, pl_txt = _prob_labels(init_fused, init_P)

    # Glow trace
    traces.append(go.Scattermapbox(
        lat=[_mlat(ft.position[1]) for ft in init_fused],
        lon=[_mlon(ft.position[0]) for ft in init_fused],
        mode="markers",
        marker=dict(size=34, color=gc, opacity=1.0),
        hoverinfo="skip", showlegend=False,
    ))

    # Fused target markers
    traces.append(go.Scattermapbox(
        lat=[_mlat(ft.position[1]) for ft in init_fused],
        lon=[_mlon(ft.position[0]) for ft in init_fused],
        mode="markers+text",
        marker=dict(size=13, color=_ft_colors(init_fused), opacity=0.97),
        text=[f"TRK-{ft.fused_index:02d}" for ft in init_fused],
        textposition="top right",
        textfont=dict(color=TEXT_COL, size=9),
        hovertext=[
            f"<b>TRACK-{ft.fused_index:02d}</b><br>"
            f"Type: {ft.target_type or '?'}<br>"
            f"Pos: ({ft.position[0]:.0f}, {ft.position[1]:.0f}) m<br>"
            f"Speed: {float(np.linalg.norm(ft.velocity_vector)):.1f} m/s  "
            f"TQ: {ft.track_quality:.2f}<br>"
            f"Conf: {'--' if ft.classification_confidence is None else f'{ft.classification_confidence:.2f}'}"
            for ft in init_fused
        ],
        hoverinfo="text", name="Fused Tracks", showlegend=True,
    ))

    # Engagement line traces (three tiers)
    traces.append(go.Scattermapbox(
        lat=lh_lat, lon=lh_lon, mode="lines",
        line=dict(color="#00e87c", width=3.0),
        opacity=0.90, hoverinfo="skip", name="P≥70%", showlegend=True,
    ))
    traces.append(go.Scattermapbox(
        lat=lm_lat, lon=lm_lon, mode="lines",
        line=dict(color="#ffd700", width=2.5),
        opacity=0.85, hoverinfo="skip", name="P 40–70%", showlegend=True,
    ))
    traces.append(go.Scattermapbox(
        lat=ll_lat, lon=ll_lon, mode="lines",
        line=dict(color="#ff4444", width=1.5),
        opacity=0.80, hoverinfo="skip", name="P<40%", showlegend=True,
    ))

    # Probability label text trace
    traces.append(go.Scattermapbox(
        lat=pl_lat, lon=pl_lon, mode="text",
        text=pl_txt,
        textfont=dict(color="white", size=11),
        hoverinfo="skip", showlegend=False,
    ))

    # ══════════════════ PLOTLY FRAMES ════════════════════════════════
    plotly_frames = []
    for fd in frames_data:
        t_val  = fd["t"]
        fused  = fd["fused_targets"]
        P      = fd["P_ml"] if fd["P_ml"] is not None else fd["P_analytical"]
        gc_f   = _glow_colors(fused, P)
        lh_lat_f, lh_lon_f = _eng_segments(fused, P, 0.70, 1.01)
        lm_lat_f, lm_lon_f = _eng_segments(fused, P, 0.40, 0.70)
        ll_lat_f, ll_lon_f = _eng_segments(fused, P, 0.00, 0.40)
        pl_lat_f, pl_lon_f, pl_txt_f = _prob_labels(fused, P)

        plotly_frames.append(go.Frame(
            data=[
                go.Scattermapbox(
                    lat=[_mlat(ft.position[1]) for ft in fused],
                    lon=[_mlon(ft.position[0]) for ft in fused],
                    marker=dict(size=34, color=gc_f, opacity=1.0),
                ),
                go.Scattermapbox(
                    lat=[_mlat(ft.position[1]) for ft in fused],
                    lon=[_mlon(ft.position[0]) for ft in fused],
                    text=[f"TRK-{ft.fused_index:02d}" for ft in fused],
                    marker=dict(size=13, color=_ft_colors(fused), opacity=0.97),
                    hovertext=[
                        f"<b>TRACK-{ft.fused_index:02d}</b><br>"
                        f"Type: {ft.target_type or '?'}<br>"
                        f"Pos: ({ft.position[0]:.0f}, {ft.position[1]:.0f}) m<br>"
                        f"Speed: {float(np.linalg.norm(ft.velocity_vector)):.1f} m/s  "
                        f"TQ: {ft.track_quality:.2f}"
                        for ft in fused
                    ],
                ),
                go.Scattermapbox(lat=lh_lat_f, lon=lh_lon_f),
                go.Scattermapbox(lat=lm_lat_f, lon=lm_lon_f),
                go.Scattermapbox(lat=ll_lat_f, lon=ll_lon_f),
                go.Scattermapbox(lat=pl_lat_f, lon=pl_lon_f, text=pl_txt_f),
            ],
            traces=[glow_idx, fused_idx, eng_hi_idx, eng_mid_idx, eng_lo_idx, prob_label_idx],
            layout=go.Layout(
                title_text=f"\U0001f3af ENGAGEMENT ASSESSMENT — T+{t_val:.0f}s",
            ),
            name=f"{t_val:.1f}",
        ))

    # ── Insert interpolated frames for smooth playback ─────────────────
    class _FP:
        __slots__ = ("position", "fused_index", "target_type")
        def __init__(self, pos, fi, tt):
            self.position    = pos
            self.fused_index = fi
            self.target_type = tt

    if len(plotly_frames) > 1:
        _N_INTERP  = 4
        _interp_out = []
        for _fi in range(len(plotly_frames) - 1):
            _interp_out.append(plotly_frames[_fi])
            _fd0, _fd1 = frames_data[_fi], frames_data[_fi + 1]
            _t0,  _t1  = _fd0["t"], _fd1["t"]
            _fus0 = _fd0["fused_targets"]
            _fus1 = _fd1["fused_targets"]
            _P0   = _fd0["P_ml"] if _fd0["P_ml"] is not None else _fd0["P_analytical"]
            _P1   = _fd1["P_ml"] if _fd1["P_ml"] is not None else _fd1["P_analytical"]
            _n_ft = min(len(_fus0), len(_fus1))
            for _k in range(1, _N_INTERP + 1):
                _a    = _k / (_N_INTERP + 1)
                _t_m  = _t0 + _a * (_t1 - _t0)
                _fus_m = [
                    _FP(
                        pos=np.array([
                            (1 - _a) * float(_fus0[_j].position[0]) + _a * float(_fus1[_j].position[0]),
                            (1 - _a) * float(_fus0[_j].position[1]) + _a * float(_fus1[_j].position[1]),
                            float(_fus0[_j].position[2]),
                        ]),
                        fi=_fus0[_j].fused_index,
                        tt=_fus0[_j].target_type,
                    )
                    for _j in range(_n_ft)
                ]
                _P_m = (1 - _a) * _P0 + _a * _P1 if (_P0 is not None and _P1 is not None and _P0.shape == _P1.shape) else _P0
                _gc_m  = _glow_colors(_fus_m, _P_m)
                _lh_la, _lh_lo = _eng_segments(_fus_m, _P_m, 0.70, 1.01)
                _lm_la, _lm_lo = _eng_segments(_fus_m, _P_m, 0.40, 0.70)
                _ll_la, _ll_lo = _eng_segments(_fus_m, _P_m, 0.00, 0.40)
                _pl_la, _pl_lo, _pl_tx = _prob_labels(_fus_m, _P_m)
                _interp_out.append(go.Frame(
                    data=[
                        go.Scattermapbox(
                            lat=[_mlat(_f.position[1]) for _f in _fus_m],
                            lon=[_mlon(_f.position[0]) for _f in _fus_m],
                            marker=dict(size=34, color=_gc_m, opacity=1.0)),
                        go.Scattermapbox(
                            lat=[_mlat(_f.position[1]) for _f in _fus_m],
                            lon=[_mlon(_f.position[0]) for _f in _fus_m],
                            text=[f"TRK-{_f.fused_index:02d}" for _f in _fus_m],
                            marker=dict(size=13,
                                        color=_ft_colors(_fus_m),
                                        opacity=0.97)),
                        go.Scattermapbox(lat=_lh_la, lon=_lh_lo),
                        go.Scattermapbox(lat=_lm_la, lon=_lm_lo),
                        go.Scattermapbox(lat=_ll_la, lon=_ll_lo),
                        go.Scattermapbox(lat=_pl_la, lon=_pl_lo, text=_pl_tx),
                    ],
                    traces=[glow_idx, fused_idx, eng_hi_idx, eng_mid_idx,
                            eng_lo_idx, prob_label_idx],
                    layout=go.Layout(
                        title_text=f"\U0001f3af ENGAGEMENT ASSESSMENT — T+{_t_m:.0f}s"),
                    name=f"{_t_m:.2f}",
                ))
        _interp_out.append(plotly_frames[-1])
        plotly_frames = _interp_out

    # Play / Pause / Slider
    has_anim    = len(plotly_frames) > 1
    updatemenus = []
    sliders_cfg = []

    if has_anim:
        updatemenus = [dict(
            type="buttons", showactive=False,
            x=0.05, xanchor="right", y=1.10, yanchor="top",
            bgcolor="#0f2340", bordercolor="#2a5080",
            font=dict(color=TEXT_COL, family="monospace"),
            buttons=[
                dict(label="▶  PLAY", method="animate",
                     args=[None, {"frame": {"duration": 150, "redraw": True},
                                  "fromcurrent": True,
                                  "transition": {"duration": 100,
                                                 "easing": "cubic-in-out"}}]),
                dict(label="⏸  PAUSE", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}]),
            ],
        )]
        sliders_cfg = [dict(
            active=0,
            currentvalue=dict(prefix="T+ ", suffix=" s",
                              font=dict(color=TEXT_COL, size=13, family="monospace"),
                              visible=True, xanchor="right"),
            pad=dict(b=10, t=55), len=0.85,
            x=0.1, y=0, xanchor="left", yanchor="top",
            bgcolor="#0f2340", bordercolor="#2a5080",
            font=dict(color=TEXT_COL, size=10, family="monospace"),
            steps=[
                dict(label=f"{fd['t']:.0f}", method="animate",
                     args=[[f"{fd['t']:.1f}"],
                           {"frame": {"duration": 100, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 80}}])
                for fd in frames_data
            ],
        )]

    layout = go.Layout(
        title=dict(
            text="\U0001f3af ENGAGEMENT ASSESSMENT — T+0s",
            font=dict(color=TEXT_COL, size=15, family="monospace"),
            x=0.5, xanchor="center",
        ),
        paper_bgcolor=PAPER,
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=_mlat(cy), lon=_mlon(cx)),
            zoom=zoom,
        ),
        legend=dict(
            bgcolor="rgba(7,14,26,0.88)", bordercolor="#2a5080",
            borderwidth=1, font=dict(color=TEXT_COL, size=10, family="monospace"),
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ),
        updatemenus=updatemenus,
        sliders=sliders_cfg,
        margin=dict(l=0, r=0, t=60, b=60),
        height=700,
    )
    return go.Figure(data=traces, layout=layout, frames=plotly_frames)



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

    # Radars placed on land near Dubai (y ≤ 7000 keeps all ≥9 km south of Gulf coast)
    # az=0° points east (threat axis), negative az tilts south-east, positive tilts north-east
    radar_defaults = [
        dict(x=-6000, y=-2000, z=50, fc=10.0, power=60, gain=30, bw=10.0, az=10.0,  el=0.0),
        dict(x=-9000, y=4000,  z=30, fc=9.0,  power=58, gain=30, bw=10.0, az=5.0,   el=0.0),
        dict(x=-3000, y=-5000, z=40, fc=10.0, power=60, gain=30, bw=10.0, az=15.0,  el=0.0),
        dict(x=-12000,y=1000,  z=50, fc=9.5,  power=55, gain=28, bw=8.0,  az=8.0,   el=0.0),
        dict(x=0,     y=6000,  z=30, fc=10.5, power=62, gain=32, bw=8.0,  az=2.0,   el=0.0),
        dict(x=-5000, y=-8000, z=60, fc=9.3,  power=57, gain=29, bw=10.0, az=20.0,  el=0.0),
        dict(x=-15000,y=3000,  z=40, fc=10.2, power=61, gain=31, bw=9.0,  az=12.0,  el=0.0),
        dict(x=-2000, y=-10000,z=50, fc=9.7,  power=59, gain=30, bw=8.0,  az=18.0,  el=0.0),
        dict(x=-18000,y=-3000, z=35, fc=10.8, power=63, gain=33, bw=10.0, az=6.0,   el=0.0),
        dict(x=4000,  y=3000,  z=45, fc=9.1,  power=56, gain=27, bw=9.0,  az=0.0,   el=0.0),
        dict(x=-8000, y=-12000,z=55, fc=10.4, power=60, gain=30, bw=10.0, az=22.0,  el=0.0),
        dict(x=-20000,y=-6000, z=30, fc=9.6,  power=58, gain=29, bw=9.0,  az=10.0,  el=0.0),
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

    # Targets start EAST of the radars, inside at least one radar's beam sector.
    # Radar 0 (-6000,-2000) az=10° bw=10°  → beam ~0°–20° from east
    # Radar 1 (-9000, 4000) az=5°  bw=10°  → beam ~0°–10° from east
    # Radar 2 (-3000,-5000) az=15° bw=10°  → beam ~10°–20° from east
    # Targets placed at angles/ranges within those cones, moving westward (neg vx).
    target_defaults = [
        dict(x=20000, y=2000,  z=2000, vx=-200, vy=-25,  vz=0, rcs=10, type="fixed_wing"),  # Radar-0 beam
        dict(x=13000, y=3000,  z=300,  vx=-60,  vy=-10,  vz=0, rcs=-5, type="drone"),       # Radar-1 beam
        dict(x=28000, y=4500,  z=1500, vx=-245, vy=-35,  vz=0, rcs=12, type="fixed_wing"),  # Radar-0/1 beam
        dict(x=9000,  y=500,   z=200,  vx=-55,  vy=-5,   vz=0, rcs=-5, type="drone"),       # Radar-1 beam
        dict(x=22000, y=-1500, z=500,  vx=-80,  vy=10,   vz=0, rcs=8,  type="helicopter"),  # Radar-2 beam
        dict(x=18000, y=5500,  z=300,  vx=-55,  vy=-15,  vz=0, rcs=-5, type="drone"),       # Radar-1 beam
        dict(x=35000, y=3500,  z=2000, vx=-280, vy=-20,  vz=0, rcs=11, type="fixed_wing"),  # Radar-0 beam
        dict(x=10000, y=-3000, z=500,  vx=-75,  vy=15,   vz=0, rcs=8,  type="helicopter"),  # Radar-2 beam
        dict(x=40000, y=5000,  z=1500, vx=-300, vy=-40,  vz=0, rcs=9,  type="fixed_wing"),  # Radar-0/1 beam
        dict(x=16000, y=1500,  z=200,  vx=-55,  vy=-10,  vz=0, rcs=-5, type="drone"),       # Radar-0 beam
        dict(x=32000, y=-2000, z=1000, vx=-250, vy=20,   vz=0, rcs=10, type="fixed_wing"),  # Radar-2 beam
        dict(x=11000, y=4000,  z=300,  vx=-60,  vy=-10,  vz=0, rcs=-5, type="drone"),       # Radar-1 beam
        dict(x=45000, y=2000,  z=2000, vx=-310, vy=-15,  vz=0, rcs=13, type="fixed_wing"),  # Radar-0 beam
        dict(x=7500,  y=-4000, z=500,  vx=-70,  vy=20,   vz=0, rcs=8,  type="helicopter"),  # Radar-2 beam
        dict(x=50000, y=6000,  z=1500, vx=-330, vy=-30,  vz=0, rcs=14, type="fixed_wing"),  # Radar-0/1 beam
        dict(x=15000, y=2500,  z=200,  vx=-55,  vy=-8,   vz=0, rcs=-5, type="drone"),       # Radar-0 beam
        dict(x=38000, y=-3000, z=1000, vx=-270, vy=15,   vz=0, rcs=9,  type="helicopter"),  # Radar-2 beam
        dict(x=24000, y=1000,  z=2000, vx=-210, vy=-20,  vz=0, rcs=12, type="fixed_wing"),  # Radar-0 beam
        dict(x=55000, y=4000,  z=1500, vx=-350, vy=-25,  vz=0, rcs=15, type="fixed_wing"),  # Radar-0 beam
        dict(x=12000, y=6000,  z=300,  vx=-55,  vy=-12,  vz=0, rcs=-5, type="drone"),       # Radar-1 beam
    ]

    _TYPE_OPTIONS = ["fixed_wing", "drone", "helicopter"]
    _TYPE_PRESETS = {
        "drone":      dict(speed=60,  rcs=-5,  z=200),
        "helicopter": dict(speed=80,  rcs=8,   z=500),
        "fixed_wing": dict(speed=220, rcs=12,  z=2000),
    }

    def _on_type_change(idx):
        new_type = st.session_state[f"t{idx}type"]
        preset = _TYPE_PRESETS[new_type]
        vx = st.session_state.get(f"t{idx}vx", -200.0)
        vy = st.session_state.get(f"t{idx}vy", 0.0)
        current_speed = np.sqrt(vx**2 + vy**2)
        if current_speed < 1.0:
            vx, vy = -1.0, 0.0
            current_speed = 1.0
        scale = preset["speed"] / current_speed
        st.session_state[f"t{idx}vx"] = round(vx * scale, 1)
        st.session_state[f"t{idx}vy"] = round(vy * scale, 1)
        st.session_state[f"t{idx}rcs"] = preset["rcs"]
        st.session_state[f"t{idx}z"]   = float(preset["z"])

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
            ttype = st.selectbox(f"Target Type##t{i}", _TYPE_OPTIONS,
                                 index=_TYPE_OPTIONS.index(d.get("type", "fixed_wing")),
                                 key=f"t{i}type",
                                 on_change=_on_type_change,
                                 args=(i,))
            target_cfgs.append(dict(x=tx, y=ty, z=tz,
                                    vx=vx, vy=vy, vz=vz, rcs=rcs, type=ttype))

    # ── Interceptor Systems ─────────────────────────────────────────
    st.subheader("Interceptor Systems")
    n_systems = st.number_input("Number of systems", min_value=1, max_value=12,
                                value=3, step=1)

    # All systems on land near Dubai (y ≤ 7000, x between -15000 and +6000)
    sys_defaults = [
        dict(name="SAM Alpha",   x=1000,  y=-1000, z=0, maxr=12000, minr=500, maxv=400, rt=5.0, sv=2),
        dict(name="SAM Beta",    x=-4000, y=3500,  z=0, maxr=15000, minr=300, maxv=600, rt=3.0, sv=3),
        dict(name="CIWS Delta",  x=3000,  y=-4000, z=0, maxr=3000,  minr=100, maxv=800, rt=1.0, sv=4),
        dict(name="SAM Gamma",   x=-1000, y=6000,  z=0, maxr=12000, minr=400, maxv=500, rt=4.0, sv=2),
        dict(name="SAM Epsilon", x=-7000, y=1000,  z=0, maxr=18000, minr=600, maxv=700, rt=6.0, sv=1),
        dict(name="CIWS Zeta",   x=5000,  y=-6000, z=0, maxr=3000,  minr=100, maxv=900, rt=0.5, sv=6),
        dict(name="SAM Eta",     x=-10000,y=5000,  z=0, maxr=20000, minr=500, maxv=550, rt=4.5, sv=2),
        dict(name="SAM Theta",   x=2000,  y=4000,  z=0, maxr=14000, minr=400, maxv=650, rt=5.5, sv=3),
        dict(name="CIWS Iota",   x=-2000, y=-5000, z=0, maxr=3000,  minr=150, maxv=850, rt=0.8, sv=5),
        dict(name="SAM Kappa",   x=-13000,y=-2000, z=0, maxr=22000, minr=700, maxv=600, rt=7.0, sv=1),
        dict(name="SAM Lambda",  x=4000,  y=1000,  z=0, maxr=12000, minr=350, maxv=480, rt=3.5, sv=3),
        dict(name="CIWS Mu",     x=-5000, y=-7000, z=0, maxr=3000,  minr=100, maxv=950, rt=0.6, sv=6),
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
    use_classification = st.checkbox("Enable Target Classification", value=True,
                                     help="Classify fused targets as drone / helicopter / fixed_wing.")
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
            target_type=c.get("type", "fixed_wing"),
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

    # If simulation has run, show animated Plotly tactical map
    if "sim_results" in st.session_state:
        sim = st.session_state["sim_results"]
        geometry = sim["frames"][selected_frame_idx]["geometry"]
        _scen_fig = build_scenario_plotly(sim, sim["systems"])
        st.plotly_chart(_scen_fig, use_container_width=True, theme=None)
        _gif_col1, _gif_col2 = st.columns([2, 1])
        with _gif_col1:
            _gif_zoom_s = st.slider(
                "GIF zoom level", min_value=0.0, max_value=6.0, value=3.0, step=0.5,
                key="gif_zoom_scenario",
                help="Higher = more zoomed in. 0 = same view as the interactive map.")
        with _gif_col2:
            st.write("")
            if st.button("🎞️ Export GIF", key="gif_scenario"):
                with st.spinner("Rendering frames — this may take ~30 s…"):
                    _gif_buf = _export_gif(_scen_fig, fps=6, zoom_boost=_gif_zoom_s)
                if _gif_buf:
                    st.download_button("⬇️ Download scenario.gif",
                                       _gif_buf.getvalue(),
                                       file_name="scenario.gif",
                                       mime="image/gif")
                else:
                    st.warning("GIF export requires kaleido and Pillow: "
                               "`pip install kaleido pillow`")
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
                target_type=tgt.target_type,
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

        # ── Propagate manually-set target type to fused targets ────────
        for ft in fused_targets:
            closest = min(frame_targets,
                          key=lambda t: np.linalg.norm(t.position[:2] - ft.position[:2]))
            ft.target_type = closest.target_type

        # ── Target classification (micro-Doppler) — overrides manual ──
        if use_classification:
            for ft in fused_targets:
                best_det = max(ft.radar_detections, key=lambda d: d["power_dB"])
                r_idx = best_det["radar_index"]
                source_det = None
                for det in per_radar_detections[r_idx]:
                    if (abs(det["range"] - best_det["range"]) < 1.0
                            and abs(det["velocity"] - best_det["velocity"]) < 0.5):
                        source_det = det
                        break
                if source_det is not None and "range_bin" in source_det:
                    rd_map   = per_radar_results[r_idx]["results"]["rd_map"]
                    vel_axis = per_radar_results[r_idx]["results"]["velocity_axis"]
                    features = extract_doppler_features(rd_map, source_det, vel_axis)
                    label, confidence = classify_target(features)
                    ft.target_type = label
                    ft.classification_confidence = confidence

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
                    "Type": ft.target_type or "—",
                    "Confidence": f"{ft.classification_confidence:.2f}" if ft.classification_confidence is not None else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # ── Classification summary ──────────────────────────────
            classified = [ft for ft in fused if ft.target_type is not None]
            if classified:
                st.subheader("Classification Summary")
                from collections import Counter
                type_counts = Counter(ft.target_type for ft in classified)
                avg_conf = {t: float(np.mean([ft.classification_confidence
                                               for ft in classified if ft.target_type == t]))
                            for t in type_counts}
                st.dataframe(pd.DataFrame([
                    {"Type": t, "Count": type_counts[t], "Avg Confidence": f"{avg_conf[t]:.2f}"}
                    for t in sorted(type_counts)
                ]), use_container_width=True)

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
        sim      = st.session_state["sim_results"]
        systems  = sim["systems"]
        frame    = sim["frames"][selected_frame_idx]
        fused    = frame["fused_targets"]
        P_ml     = frame["P_ml"]
        P_ana    = frame["P_analytical"]
        P        = P_ml if P_ml is not None else P_ana
        prob_label = "ML" if P_ml is not None else "Analytical"

        # ── Animated Plotly recommendation map ───────────────────────
        _rec_fig = build_recommendation_plotly(sim, systems)
        st.plotly_chart(_rec_fig, use_container_width=True, theme=None)
        _gif_col1, _gif_col2 = st.columns([2, 1])
        with _gif_col1:
            _gif_zoom_r = st.slider(
                "GIF zoom level", min_value=0.0, max_value=6.0, value=3.0, step=0.5,
                key="gif_zoom_recommend",
                help="Higher = more zoomed in. 0 = same view as the interactive map.")
        with _gif_col2:
            st.write("")
            if st.button("🎞️ Export GIF", key="gif_recommend"):
                with st.spinner("Rendering frames — this may take ~30 s…"):
                    _gif_buf = _export_gif(_rec_fig, fps=6, zoom_boost=_gif_zoom_r)
                if _gif_buf:
                    st.download_button("⬇️ Download recommendation.gif",
                                       _gif_buf.getvalue(),
                                       file_name="recommendation.gif",
                                       mime="image/gif")
                else:
                    st.warning("GIF export requires kaleido and Pillow: "
                               "`pip install kaleido pillow`")

        # ── Summary table (driven by Streamlit time slider) ──────────
        if not fused:
            st.warning("No fused targets detected at this timestep.")
        else:
            st.caption(
                f"Summary at t = {frame['t']:.1f} s — source: **{prob_label}**"
            )
            recommendations = []
            for j, ft in enumerate(fused):
                if j < P.shape[1]:
                    best_idx  = int(np.argmax(P[:, j]))
                    best_prob = float(P[best_idx, j])
                    recommendations.append({
                        "Target": f"FT{ft.fused_index}",
                        "Position": f"({ft.position[0]:.0f}, {ft.position[1]:.0f})",
                        "Type": ft.target_type or "\u2014",
                        "Best Interceptor": systems[best_idx].name if best_prob > 0 else "\u2014",
                        "P(intercept)": f"{best_prob * 100:.1f}%",
                    })
            if recommendations:
                st.subheader("Recommendation Summary")
                st.dataframe(pd.DataFrame(recommendations), use_container_width=True)
