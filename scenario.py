import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


C = 3e8  # speed of light [m/s]


@dataclass
class Radar:
    position: np.ndarray           # (x, y, z) in meters
    fc: float                      # carrier frequency [Hz]
    tx_power_dBm: float = 60.0     # transmit power [dBm]
    antenna_gain_dB: float = 30.0  # peak antenna gain [dB]
    antenna_beamwidth: float = 5.0 # 3-dB beamwidth [deg]
    look_azimuth: float = 0.0      # boresight azimuth [deg] from +x axis
    look_elevation: float = 0.0    # boresight elevation [deg] from x-y plane

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)

    @property
    def wavelength(self) -> float:
        return C / self.fc

    @property
    def tx_power_watts(self) -> float:
        return 10 ** ((self.tx_power_dBm - 30) / 10)

    @property
    def antenna_gain_linear(self) -> float:
        return 10 ** (self.antenna_gain_dB / 10)

    def beam_pattern(self, azimuth_deg: float, elevation_deg: float) -> float:
        """Compute antenna gain [dB] at given azimuth/elevation using sinc² pattern."""
        az_off = azimuth_deg - self.look_azimuth
        el_off = elevation_deg - self.look_elevation
        total_off = np.sqrt(az_off**2 + el_off**2)

        bw = self.antenna_beamwidth
        if total_off < 1e-10:
            return self.antenna_gain_dB

        # sinc² pattern: G(theta) = G_peak * sinc²(0.886 * theta / bw)
        u = 0.886 * total_off / bw
        sinc_val = np.sinc(u)  # numpy sinc includes the pi factor
        gain_dB = self.antenna_gain_dB + 20 * np.log10(np.abs(sinc_val) + 1e-30)
        return gain_dB


@dataclass
class Target:
    position: np.ndarray   # (x, y, z) in meters
    velocity: np.ndarray   # (vx, vy, vz) in m/s
    rcs_dbsm: float        # radar cross section [dBsm]

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)

    @property
    def rcs_linear(self) -> float:
        return 10 ** (self.rcs_dbsm / 10)


@dataclass
class TargetGeometry:
    """Computed geometric relationship between radar and a single target."""
    radar_index: int
    target_index: int
    range_m: float
    radial_velocity: float    # m/s, positive = moving away
    azimuth_deg: float
    elevation_deg: float
    antenna_gain_dB: float
    rcs_dbsm: float
    rcs_linear: float


class Scenario:
    def __init__(self, radars: List[Radar], targets: List[Target]):
        if isinstance(radars, Radar):
            radars = [radars]
        self.radars = radars
        self.targets = targets

    def compute_geometry(self) -> Dict[int, List[TargetGeometry]]:
        """Compute geometric parameters for each target relative to each radar.

        Returns a dict keyed by radar index, each value a list of TargetGeometry.
        """
        all_geometry = {}
        for r_idx, radar in enumerate(self.radars):
            results = []
            for i, tgt in enumerate(self.targets):
                delta = tgt.position - radar.position
                R = np.linalg.norm(delta)

                # direction unit vector
                u = delta / R if R > 0 else np.array([1.0, 0.0, 0.0])

                # radial velocity (positive = receding)
                v_r = np.dot(tgt.velocity, u)

                # azimuth: angle in x-y plane from +x axis [deg]
                azimuth = np.degrees(np.arctan2(u[1], u[0]))

                # elevation: angle above x-y plane [deg]
                elevation = np.degrees(np.arcsin(np.clip(u[2], -1, 1)))

                # antenna gain at target direction
                gain_dB = radar.beam_pattern(azimuth, elevation)

                results.append(TargetGeometry(
                    radar_index=r_idx,
                    target_index=i,
                    range_m=R,
                    radial_velocity=v_r,
                    azimuth_deg=azimuth,
                    elevation_deg=elevation,
                    antenna_gain_dB=gain_dB,
                    rcs_dbsm=tgt.rcs_dbsm,
                    rcs_linear=tgt.rcs_linear,
                ))
            all_geometry[r_idx] = results
        return all_geometry

    def summary(self, geometry: Dict[int, List[TargetGeometry]] = None):
        """Print a table summarising the scenario geometry for each radar."""
        if geometry is None:
            geometry = self.compute_geometry()

        for r_idx, radar in enumerate(self.radars):
            geom_list = geometry[r_idx]
            print("=" * 80)
            print(f"  Radar {r_idx}: pos={radar.position}, fc={radar.fc/1e9:.2f} GHz, "
                  f"Pt={radar.tx_power_dBm:.0f} dBm, G={radar.antenna_gain_dB:.1f} dB")
            print("-" * 80)
            print(f"  {'Tgt':>3s}  {'Range [m]':>10s}  {'Vr [m/s]':>10s}  {'Az [°]':>8s}  "
                  f"{'El [°]':>8s}  {'Gain [dB]':>10s}  {'RCS [dBsm]':>11s}")
            print("-" * 80)
            for g in geom_list:
                print(f"  {g.target_index:3d}  {g.range_m:10.1f}  {g.radial_velocity:10.2f}  "
                      f"{g.azimuth_deg:8.2f}  {g.elevation_deg:8.2f}  "
                      f"{g.antenna_gain_dB:10.2f}  {g.rcs_dbsm:11.1f}")
            print("=" * 80)

    def plot_scenario(self, geometry: Dict[int, List[TargetGeometry]] = None,
                      interceptor_systems=None,
                      save_path: str = None):
        """Plot a 2D top-down (x-y) view of the scenario with all radars.

        Parameters
        ----------
        geometry : dict keyed by radar index
        interceptor_systems : optional list of InterceptorSystem (with .position,
            .name, .min_range, .max_range)
        save_path : file path for saving the figure
        """
        if geometry is None:
            geometry = self.compute_geometry()

        fig, ax = plt.subplots(figsize=(13, 9))

        radar_colors = plt.cm.Set1(np.linspace(0, 1, max(len(self.radars), 3)))

        # collect all ranges for scaling
        all_ranges = [g.range_m for geom_list in geometry.values()
                      for g in geom_list]
        max_r = max(all_ranges) * 1.2 if all_ranges else 1000

        # ── Draw each radar and its beam ─────────────────────────────
        for r_idx, radar in enumerate(self.radars):
            color = radar_colors[r_idx]
            rx, ry = radar.position[0], radar.position[1]
            ax.plot(rx, ry, "s", color=color, markersize=12,
                    label=f"Radar {r_idx} ({radar.fc/1e9:.1f} GHz)")
            ax.annotate(f"R{r_idx}", (rx, ry),
                        textcoords="offset points", xytext=(-15, -12),
                        fontsize=7, color=color, fontweight="bold")

            # beam sector (filled wedge)
            look_az_rad = np.radians(radar.look_azimuth)
            bw_rad = np.radians(radar.antenna_beamwidth)
            angles = np.linspace(look_az_rad - bw_rad / 2,
                                 look_az_rad + bw_rad / 2, 40)
            wedge_x = np.concatenate([[rx], rx + max_r * np.cos(angles), [rx]])
            wedge_y = np.concatenate([[ry], ry + max_r * np.sin(angles), [ry]])
            ax.fill(wedge_x, wedge_y, color=color, alpha=0.06)
            for sign in [-1, 1]:
                angle = look_az_rad + sign * bw_rad / 2
                ax.plot([rx, rx + max_r * np.cos(angle)],
                        [ry, ry + max_r * np.sin(angle)],
                        "--", color=color, alpha=0.3, linewidth=1)

        # ── Targets ──────────────────────────────────────────────────
        for i, tgt in enumerate(self.targets):
            tx, ty = tgt.position[0], tgt.position[1]
            ax.plot(tx, ty, "bo", markersize=8,
                    label="Target" if i == 0 else None)
            ax.annotate(f"T{i}\n({tx:.0f}, {ty:.0f})",
                        (tx, ty), textcoords="offset points", xytext=(10, 5),
                        fontsize=7, color="navy")
            # velocity arrow
            vel_mag = np.linalg.norm(tgt.velocity[:2])
            scale = max_r * 0.05 / (vel_mag + 1e-6)
            ax.annotate("", xy=(tx + tgt.velocity[0] * scale,
                                ty + tgt.velocity[1] * scale),
                        xytext=(tx, ty),
                        arrowprops=dict(arrowstyle="->", color="green", lw=1.5))
            ax.annotate(f"{vel_mag:.0f} m/s",
                        (tx + tgt.velocity[0] * scale * 0.5,
                         ty + tgt.velocity[1] * scale * 0.5),
                        fontsize=6, color="green", alpha=0.7)

        # ── Interceptor systems ──────────────────────────────────────
        if interceptor_systems:
            intc_colors = plt.cm.Dark2(np.linspace(0, 1,
                                       max(len(interceptor_systems), 3)))
            for s_idx, sys in enumerate(interceptor_systems):
                sx, sy = sys.position[0], sys.position[1]
                color = intc_colors[s_idx]
                ax.plot(sx, sy, "^", color=color, markersize=10,
                        label=sys.name)
                ax.annotate(f"{sys.name}",
                            (sx, sy), textcoords="offset points",
                            xytext=(8, -12), fontsize=6, color=color,
                            fontweight="bold")

                # min/max range rings
                theta = np.linspace(0, 2 * np.pi, 120)
                ax.plot(sx + sys.max_range * np.cos(theta),
                        sy + sys.max_range * np.sin(theta),
                        "-", color=color, alpha=0.25, linewidth=0.8)
                ax.plot(sx + sys.min_range * np.cos(theta),
                        sy + sys.min_range * np.sin(theta),
                        ":", color=color, alpha=0.2, linewidth=0.6)

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("Scenario - Top-Down View (Multi-Radar)")
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Scenario plot saved to {save_path}")
        return fig, ax
