"""
interceptor.py — Intercept Probability Blackbox

Takes a list of InterceptorSystem definitions (each with a position) and
a list of FusedTarget objects (from multi-radar fusion) and returns an
intercept probability for every (system, target) pair.

The internal calculation uses interceptor-to-target distance and closing
speed (derived from Cartesian position/velocity estimates).  The body of
InterceptBlackbox._compute() can be swapped at any time without touching
anything else in the project.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fusion import FusedTarget
from feature_extraction import extract_from_system_target, FEATURE_NAMES


# ── System definition ────────────────────────────────────────────────

@dataclass
class InterceptorSystem:
    name: str
    position: np.ndarray            # (x, y, z) position of the system [m]
    max_range: float                # maximum engagement range [m]
    min_range: float                # minimum engagement range [m] (dead zone)
    max_target_velocity: float      # maximum closing speed the system can handle [m/s]
    reaction_time: float            # time from detection to launch [s]
    salvo_size: int = 1             # number of interceptors fired per engagement

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)


# ── Blackbox ─────────────────────────────────────────────────────────

class InterceptBlackbox:
    """Blackbox that produces intercept probabilities.

    Input
    -----
    systems : list of InterceptorSystem
    targets : list of FusedTarget (from multi-radar fusion)

    Output
    ------
    probability_matrix : np.ndarray, shape (N_systems, N_targets)
        probability[i, j] = P(system i intercepts target j)
    """

    def __init__(self, systems: List[InterceptorSystem],
                 use_ml: bool = False,
                 model_path: str = "xgb_intercept_model.json"):
        self.systems = systems
        self.use_ml = use_ml
        self._model = None

        if use_ml:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"XGBoost model not found at '{model_path}'. "
                    f"Run train_model.py first, or set use_ml=False."
                )
            from xgboost import XGBClassifier
            self._model = XGBClassifier()
            self._model.load_model(model_path)
            print(f"  [InterceptBlackbox] Loaded ML model from {model_path}")

    # ── PUBLIC API ───────────────────────────────────────────────────

    def evaluate(self, targets: List[FusedTarget]) -> np.ndarray:
        """Return the (N_systems x N_targets) intercept probability matrix."""
        n_sys = len(self.systems)
        n_tgt = len(targets)
        P = np.zeros((n_sys, n_tgt))

        for i, sys in enumerate(self.systems):
            for j, ft in enumerate(targets):
                tgt_dict = {
                    "position": ft.position,
                    "velocity_vector": ft.velocity_vector,
                    "n_radars": ft.n_radars,
                    "track_quality": ft.track_quality,
                    "power_dB": ft.power_dB,
                    "radar_detections": ft.radar_detections,
                    "position_method": ft.position_method,
                }
                P[i, j] = self._compute(sys, tgt_dict)

        return P

    # ── BLACKBOX INTERNALS ──────────────────────────────────────────

    def _compute(self, system: InterceptorSystem,
                 target: Dict[str, Any]) -> float:
        """Route to ML or analytical model based on use_ml flag."""
        if self.use_ml and self._model is not None:
            return self._compute_ml(system, target)
        return self._compute_analytical(system, target)

    def _compute_ml(self, system: InterceptorSystem,
                    target: Dict[str, Any]) -> float:
        """Intercept probability via the trained XGBoost model."""
        # Build a lightweight object that has the fields extract_from_system_target expects
        class _FT:
            pass
        ft = _FT()
        ft.position = np.asarray(target["position"])
        ft.velocity_vector = np.asarray(target["velocity_vector"])
        ft.track_quality = target.get("track_quality", 0.5)
        ft.n_radars = target.get("n_radars", 1)
        ft.power_dB = target.get("power_dB", 0.0)
        ft.position_method = target.get("position_method", "boresight_approx")

        features = extract_from_system_target(system, ft).reshape(1, -1)
        prob = self._model.predict_proba(features)[0, 1]
        return float(np.clip(prob, 0.0, 1.0))

    def _compute_analytical(self, system: InterceptorSystem,
                            target: Dict[str, Any]) -> float:
        """Analytical intercept probability (original formula).

        Uses Cartesian position & velocity estimates from multi-radar fusion:
          - Range check: interceptor-to-target Euclidean distance
          - Velocity check: closing speed toward the interceptor
          - Track quality factor: multi-radar coverage improves probability
        """
        tgt_pos = np.asarray(target["position"])
        tgt_vel = np.asarray(target["velocity_vector"])

        # ── Interceptor-to-target distance ───────────────────────────
        delta = tgt_pos - system.position
        R = np.linalg.norm(delta)

        # ── Closing speed toward interceptor ─────────────────────────
        if R > 0:
            u_to_interceptor = -delta / R   # unit vector: target → interceptor
            closing_speed = np.dot(tgt_vel, u_to_interceptor)
        else:
            closing_speed = 0.0

        V = abs(closing_speed)

        # ── Envelope check ───────────────────────────────────────────
        in_range = system.min_range <= R <= system.max_range
        in_speed = V <= system.max_target_velocity

        if not in_range or not in_speed:
            return 0.0

        # ── Range factor: linear falloff from min to max range ───────
        span = system.max_range - system.min_range
        range_factor = 1.0 - (R - system.min_range) / span

        # ── Speed factor: linear falloff ─────────────────────────────
        speed_factor = 1.0 - V / system.max_target_velocity

        # ── Track quality factor ─────────────────────────────────────
        quality = target.get("track_quality", 0.5)
        quality_factor = np.sqrt(max(quality, 0.1))  # sqrt scaling, floor at 0.1

        # ── Single-shot probability ──────────────────────────────────
        p_single = np.cbrt(range_factor * speed_factor * quality_factor)

        # ── Multi-round salvo (independent shots) ────────────────────
        p_intercept = 1.0 - (1.0 - p_single) ** system.salvo_size

        return float(np.clip(p_intercept, 0.0, 1.0))


# ── Display helpers ──────────────────────────────────────────────────

def print_intercept_table(systems: List[InterceptorSystem],
                          targets: List[FusedTarget],
                          P: np.ndarray):
    """Print a formatted probability matrix to the console."""
    n_sys, n_tgt = P.shape

    tgt_labels = [f"  FT{j}  " for j in range(n_tgt)]
    col_w = 10

    header  = f"  {'System':<20s}" + "".join(f"{lbl:>{col_w}s}" for lbl in tgt_labels)
    divider = "  " + "-" * (20 + col_w * n_tgt)

    print(header)
    print(divider)
    for i, sys in enumerate(systems):
        row = f"  {sys.name:<20s}"
        for j in range(n_tgt):
            pct = P[i, j] * 100
            row += f"{pct:>{col_w}.1f}%"
        print(row)
    print(divider)

    # Detail per target
    print("\n  Per-target detail:")
    for j, ft in enumerate(targets):
        radar_str = ",".join(str(d["radar_index"]) for d in ft.radar_detections)
        print(f"    FT{j}: pos=({ft.position[0]:.0f}, {ft.position[1]:.0f}, {ft.position[2]:.0f})m, "
              f"|v|={np.linalg.norm(ft.velocity_vector):.1f}m/s, "
              f"radars=[{radar_str}], method={ft.position_method}")

        # interceptor-to-target distances and closing speeds
        for i, sys in enumerate(systems):
            delta = ft.position - sys.position
            dist = np.linalg.norm(delta)
            if dist > 0:
                u = -delta / dist
                cs = np.dot(ft.velocity_vector, u)
            else:
                cs = 0.0
            pct = P[i, j] * 100
            print(f"      {sys.name:<18s}: dist={dist:.0f}m, closing={cs:.1f}m/s, P={pct:.1f}%")

    # Best system per target
    print("\n  Best system per target:")
    for j, ft in enumerate(targets):
        col = P[:, j]
        if col.max() == 0:
            print(f"    FT{j}: no system in envelope")
        else:
            best_i = int(np.argmax(col))
            print(f"    FT{j}: {systems[best_i].name}  ({col[best_i]*100:.1f}%)")


def plot_intercept_matrix(systems: List[InterceptorSystem],
                          targets: List[FusedTarget],
                          P: np.ndarray,
                          save_path: str = "intercept_matrix.png"):
    """Plot the intercept probability matrix as a heatmap."""
    n_sys, n_tgt = P.shape

    fig, ax = plt.subplots(figsize=(max(6, n_tgt * 1.6), max(4, n_sys * 0.9)))

    im = ax.imshow(P * 100, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="Intercept probability [%]")

    tgt_labels = []
    for j, ft in enumerate(targets):
        dist_strs = []
        for sys in systems:
            d = np.linalg.norm(ft.position - sys.position)
            dist_strs.append(f"{d:.0f}m")
        radar_str = ",".join(str(d["radar_index"]) for d in ft.radar_detections)
        tgt_labels.append(f"FT{j}\nR=[{radar_str}]\n{ft.position_method[:5]}")

    sys_labels = [f"{s.name}\n({s.position[0]:.0f},{s.position[1]:.0f})" for s in systems]

    ax.set_xticks(range(n_tgt))
    ax.set_xticklabels(tgt_labels, fontsize=8)
    ax.set_yticks(range(n_sys))
    ax.set_yticklabels(sys_labels, fontsize=8)

    # annotate cells
    for i in range(n_sys):
        for j in range(n_tgt):
            pct = P[i, j] * 100
            color = "white" if pct < 30 or pct > 70 else "black"
            ax.text(j, i, f"{pct:.1f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_title("Intercept Probability Matrix\n(system vs. fused target)")
    ax.set_xlabel("Fused target (multi-radar)")
    ax.set_ylabel("Interceptor system")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Intercept matrix plot saved to {save_path}")
    return fig, ax
