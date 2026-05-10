"""
interceptor.py — Intercept Probability Blackbox

Takes a list of InterceptorSystem definitions and a list of target
estimations (as produced by the radar processing pipeline) and returns
an intercept probability for every (system, target) pair.

The internal calculation is a placeholder.  The interface is the
contract: swap the body of InterceptBlackbox._compute() at any time
without touching anything else in the project.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── System definition ────────────────────────────────────────────────

@dataclass
class InterceptorSystem:
    name: str
    max_range: float            # maximum engagement range [m]
    min_range: float            # minimum engagement range [m] (dead zone)
    max_target_velocity: float  # maximum target speed the system can handle [m/s]
    reaction_time: float        # time from detection to launch [s]
    salvo_size: int = 1         # number of interceptors fired per engagement
    # Any extra parameters specific to this system go here as needed


# ── Blackbox ─────────────────────────────────────────────────────────

class InterceptBlackbox:
    """Blackbox that produces intercept probabilities.

    Input
    -----
    systems : list of InterceptorSystem
    targets : list of dicts with at least keys
                  'range'     – estimated target range [m]
                  'velocity'  – estimated target radial velocity [m/s]
                  'power_dB'  – peak power in the RD map [dB]
              (i.e. exactly what RadarProcessor.estimate_targets() returns)

    Output
    ------
    probability_matrix : np.ndarray, shape (N_systems, N_targets)
        probability[i, j] = P(system i intercepts target j)
    """

    def __init__(self, systems: List[InterceptorSystem]):
        self.systems = systems

    # ── PUBLIC API ───────────────────────────────────────────────────

    def evaluate(self, targets: List[Dict[str, Any]]) -> np.ndarray:
        """Return the (N_systems x N_targets) intercept probability matrix."""
        n_sys = len(self.systems)
        n_tgt = len(targets)
        P = np.zeros((n_sys, n_tgt))

        for i, sys in enumerate(self.systems):
            for j, tgt in enumerate(targets):
                P[i, j] = self._compute(sys, tgt)

        return P

    # ── BLACKBOX INTERNALS (placeholder) ────────────────────────────

    def _compute(self, system: InterceptorSystem,
                 target: Dict[str, Any]) -> float:
        """Placeholder intercept probability calculation.

        Replace this method with a real engagement model without
        changing any other part of the code.

        Current placeholder logic (illustrative only):
          - Target within engagement envelope?  → base probability
          - Degrade with range (further = harder)
          - Degrade with target speed (faster = harder)
          - Boost slightly for multi-round salvos
        """
        R   = target["range"]          # [m]
        V   = abs(target["velocity"])  # [m/s], use magnitude

        # ── Envelope check ───────────────────────────────────────────
        in_range = system.min_range <= R <= system.max_range
        in_speed = V <= system.max_target_velocity

        if not in_range or not in_speed:
            return 0.0

        # ── Range factor: linear falloff from min to max range ───────
        span = system.max_range - system.min_range
        range_factor = 1.0 - (R - system.min_range) / span   # 1 at min, 0 at max

        # ── Speed factor: linear falloff ─────────────────────────────
        speed_factor = 1.0 - V / system.max_target_velocity   # 1 at 0, 0 at max

        # ── Single-shot probability (geometric mean of factors) ──────
        p_single = np.sqrt(range_factor * speed_factor)

        # ── Multi-round salvo (independent shots) ────────────────────
        p_intercept = 1.0 - (1.0 - p_single) ** system.salvo_size

        return float(np.clip(p_intercept, 0.0, 1.0))


# ── Display helpers ──────────────────────────────────────────────────

def print_intercept_table(systems: List[InterceptorSystem],
                          targets: List[Dict[str, Any]],
                          P: np.ndarray):
    """Print a formatted probability matrix to the console."""
    n_sys, n_tgt = P.shape

    tgt_labels = [f"  T{j}  " for j in range(n_tgt)]
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

    # Best system per target
    print("\n  Best system per target:")
    for j in range(n_tgt):
        col = P[:, j]
        if col.max() == 0:
            print(f"    T{j}: no system in envelope")
        else:
            best_i = int(np.argmax(col))
            print(f"    T{j} (R={targets[j]['range']:.0f}m, "
                  f"V={targets[j]['velocity']:.1f}m/s): "
                  f"{systems[best_i].name}  ({col[best_i]*100:.1f}%)")


def plot_intercept_matrix(systems: List[InterceptorSystem],
                          targets: List[Dict[str, Any]],
                          P: np.ndarray,
                          save_path: str = "intercept_matrix.png"):
    """Plot the intercept probability matrix as a heatmap."""
    n_sys, n_tgt = P.shape

    fig, ax = plt.subplots(figsize=(max(6, n_tgt * 1.4), max(4, n_sys * 0.9)))

    im = ax.imshow(P * 100, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="Intercept probability [%]")

    tgt_labels = [f"T{j}\n{targets[j]['range']:.0f}m\n{targets[j]['velocity']:.1f}m/s"
                  for j in range(n_tgt)]
    sys_labels  = [s.name for s in systems]

    ax.set_xticks(range(n_tgt))
    ax.set_xticklabels(tgt_labels, fontsize=9)
    ax.set_yticks(range(n_sys))
    ax.set_yticklabels(sys_labels, fontsize=9)

    # annotate cells
    for i in range(n_sys):
        for j in range(n_tgt):
            pct = P[i, j] * 100
            color = "white" if pct < 30 or pct > 70 else "black"
            ax.text(j, i, f"{pct:.1f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_title("Intercept Probability Matrix\n(system vs. estimated target)")
    ax.set_xlabel("Radar-estimated target")
    ax.set_ylabel("Interceptor system")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Intercept matrix plot saved to {save_path}")
    return fig, ax
