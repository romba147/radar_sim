import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
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
    target_index: int
    range_m: float
    radial_velocity: float    # m/s, positive = moving away
    azimuth_deg: float
    elevation_deg: float
    antenna_gain_dB: float
    rcs_dbsm: float
    rcs_linear: float


class Scenario:
    def __init__(self, radar: Radar, targets: List[Target]):
        self.radar = radar
        self.targets = targets

    def compute_geometry(self) -> List[TargetGeometry]:
        """Compute geometric parameters for each target relative to the radar."""
        results = []
        for i, tgt in enumerate(self.targets):
            delta = tgt.position - self.radar.position
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
            gain_dB = self.radar.beam_pattern(azimuth, elevation)

            results.append(TargetGeometry(
                target_index=i,
                range_m=R,
                radial_velocity=v_r,
                azimuth_deg=azimuth,
                elevation_deg=elevation,
                antenna_gain_dB=gain_dB,
                rcs_dbsm=tgt.rcs_dbsm,
                rcs_linear=tgt.rcs_linear,
            ))
        return results

    def summary(self, geometry: List[TargetGeometry] = None):
        """Print a table summarising the scenario geometry."""
        if geometry is None:
            geometry = self.compute_geometry()

        print("=" * 80)
        print(f"  Radar  : pos={self.radar.position}, fc={self.radar.fc/1e9:.2f} GHz, "
              f"Pt={self.radar.tx_power_dBm:.0f} dBm, G={self.radar.antenna_gain_dB:.1f} dB")
        print("-" * 80)
        print(f"  {'Tgt':>3s}  {'Range [m]':>10s}  {'Vr [m/s]':>10s}  {'Az [°]':>8s}  "
              f"{'El [°]':>8s}  {'Gain [dB]':>10s}  {'RCS [dBsm]':>11s}")
        print("-" * 80)
        for g in geometry:
            print(f"  {g.target_index:3d}  {g.range_m:10.1f}  {g.radial_velocity:10.2f}  "
                  f"{g.azimuth_deg:8.2f}  {g.elevation_deg:8.2f}  "
                  f"{g.antenna_gain_dB:10.2f}  {g.rcs_dbsm:11.1f}")
        print("=" * 80)

    def plot_scenario(self, geometry: List[TargetGeometry] = None,
                      save_path: str = None):
        """Plot a 2D top-down (x-y) view of the scenario."""
        if geometry is None:
            geometry = self.compute_geometry()

        fig, ax = plt.subplots(figsize=(9, 7))

        # radar
        rx, ry = self.radar.position[0], self.radar.position[1]
        ax.plot(rx, ry, "rs", markersize=12, label="Radar")

        # draw beam sector (approximate)
        look_az_rad = np.radians(self.radar.look_azimuth)
        bw_rad = np.radians(self.radar.antenna_beamwidth)
        max_r = max(g.range_m for g in geometry) * 1.2

        for sign in [-1, 1]:
            angle = look_az_rad + sign * bw_rad / 2
            ax.plot([rx, rx + max_r * np.cos(angle)],
                    [ry, ry + max_r * np.sin(angle)],
                    "r--", alpha=0.3, linewidth=1)

        # targets
        for tgt, g in zip(self.targets, geometry):
            tx, ty = tgt.position[0], tgt.position[1]
            ax.plot(tx, ty, "bo", markersize=8)
            ax.annotate(f"T{g.target_index}\n{g.range_m:.0f}m\n{g.radial_velocity:.1f}m/s",
                        (tx, ty), textcoords="offset points", xytext=(10, 5),
                        fontsize=8, color="navy")
            # velocity arrow
            scale = max_r * 0.05 / (np.linalg.norm(tgt.velocity[:2]) + 1e-6)
            ax.annotate("", xy=(tx + tgt.velocity[0] * scale,
                                ty + tgt.velocity[1] * scale),
                        xytext=(tx, ty),
                        arrowprops=dict(arrowstyle="->", color="green", lw=1.5))

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("Scenario - Top-Down View")
        ax.set_aspect("equal")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Scenario plot saved to {save_path}")
        return fig, ax
