import numpy as np
import matplotlib
matplotlib.use("Agg")
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt

from scenario import C, TargetGeometry
from signal_gen import WaveformConfig


class WindowType(Enum):
    RECTANGULAR = "rectangular"
    HANNING = "hanning"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    KAISER = "kaiser"


@dataclass
class ProcessingConfig:
    range_window: WindowType = WindowType.HANNING
    doppler_window: WindowType = WindowType.HANNING
    cfar_guard_cells: int = 4
    cfar_training_cells: int = 16
    cfar_threshold_factor: float = 5.0
    mti_enabled: bool = False
    mti_order: int = 1          # 1 = 2-pulse canceller, 2 = 3-pulse canceller
    kaiser_beta: float = 6.0    # beta parameter for Kaiser window


def _make_window(window_type: WindowType, length: int,
                 kaiser_beta: float = 6.0) -> np.ndarray:
    """Create a window of the given type and length."""
    if window_type == WindowType.RECTANGULAR:
        return np.ones(length)
    elif window_type == WindowType.HANNING:
        return np.hanning(length)
    elif window_type == WindowType.HAMMING:
        return np.hamming(length)
    elif window_type == WindowType.BLACKMAN:
        return np.blackman(length)
    elif window_type == WindowType.KAISER:
        return np.kaiser(length, kaiser_beta)
    else:
        return np.ones(length)


class RadarProcessor:
    def __init__(self, waveform: WaveformConfig, config: ProcessingConfig):
        self.wf = waveform
        self.cfg = config

    def matched_filter(self, rx_signal: np.ndarray,
                       tx_ref: np.ndarray) -> np.ndarray:
        """Pulse compression via frequency-domain matched filtering.

        For each pulse, multiply the FFT of the received signal by the
        conjugate of the FFT of the reference waveform.
        """
        N = rx_signal.shape[0]
        ref_fft = np.fft.fft(tx_ref, n=N)
        ref_conj = np.conj(ref_fft).reshape(-1, 1)

        rx_fft = np.fft.fft(rx_signal, axis=0)
        compressed = np.fft.ifft(rx_fft * ref_conj, axis=0)
        return compressed

    def mti_filter(self, rx_signal: np.ndarray) -> np.ndarray:
        """MTI clutter cancellation via pulse-to-pulse subtraction.

        Order 1 (2-pulse canceller): y[m] = x[m] - x[m-1]
        Order 2 (3-pulse canceller): y[m] = x[m] - 2*x[m-1] + x[m-2]
        """
        order = self.cfg.mti_order
        if order == 1:
            return rx_signal[:, 1:] - rx_signal[:, :-1]
        elif order == 2:
            return (rx_signal[:, 2:]
                    - 2 * rx_signal[:, 1:-1]
                    + rx_signal[:, :-2])
        else:
            return rx_signal[:, 1:] - rx_signal[:, :-1]

    def range_doppler_map(self, rx_signal: np.ndarray) -> np.ndarray:
        """Compute the Range-Doppler map with windowed FFTs."""
        N, M = rx_signal.shape

        # range window (along fast-time)
        r_win = _make_window(self.cfg.range_window, N,
                             self.cfg.kaiser_beta).reshape(-1, 1)
        range_fft = np.fft.fft(rx_signal * r_win, axis=0)

        # doppler window (along slow-time)
        d_win = _make_window(self.cfg.doppler_window, M,
                             self.cfg.kaiser_beta).reshape(1, -1)
        rd_map = np.fft.fftshift(
            np.fft.fft(range_fft * d_win, axis=1), axes=1
        )
        return rd_map

    def cfar_detection(self, rd_map_mag: np.ndarray) -> Dict[str, np.ndarray]:
        """2-D Cell-Averaging CFAR detection.

        Parameters
        ----------
        rd_map_mag : 2-D array of magnitude (linear) values.

        Returns
        -------
        dict with keys:
            'detection_mask' : bool array, True where target detected
            'threshold_map'  : the adaptive threshold at each cell
        """
        Nr, Nd = rd_map_mag.shape
        guard = self.cfg.cfar_guard_cells
        train = self.cfg.cfar_training_cells
        alpha = self.cfg.cfar_threshold_factor

        threshold_map = np.zeros_like(rd_map_mag)
        detection_mask = np.zeros_like(rd_map_mag, dtype=bool)

        margin_r = guard + train
        margin_d = guard + train

        for i in range(margin_r, Nr - margin_r):
            for j in range(margin_d, Nd - margin_d):
                # training region: ring around CUT, excluding guard cells
                region = rd_map_mag[i - margin_r:i + margin_r + 1,
                                    j - margin_d:j + margin_d + 1].copy()
                # zero-out guard + CUT
                gi, gj = margin_r, margin_d  # CUT position in the region
                region[gi - guard:gi + guard + 1,
                       gj - guard:gj + guard + 1] = 0.0

                n_training = region.size - (2 * guard + 1)**2
                if n_training > 0:
                    noise_level = region.sum() / n_training
                else:
                    noise_level = 0.0

                threshold_map[i, j] = alpha * noise_level
                if rd_map_mag[i, j] > threshold_map[i, j]:
                    detection_mask[i, j] = True

        return {"detection_mask": detection_mask, "threshold_map": threshold_map}

    def estimate_targets(self, rd_map: np.ndarray,
                         detection_mask: np.ndarray,
                         range_axis: np.ndarray,
                         velocity_axis: np.ndarray) -> List[Dict[str, float]]:
        """Extract target estimates from detection mask.

        Groups adjacent detections into clusters and picks the peak in each.
        """
        rd_mag = np.abs(rd_map)
        visited = np.zeros_like(detection_mask, dtype=bool)
        detections = []

        Nr, Nd = detection_mask.shape

        for i in range(Nr):
            for j in range(Nd):
                if detection_mask[i, j] and not visited[i, j]:
                    # flood-fill cluster
                    cluster = []
                    stack = [(i, j)]
                    while stack:
                        ci, cj = stack.pop()
                        if (0 <= ci < Nr and 0 <= cj < Nd
                                and detection_mask[ci, cj]
                                and not visited[ci, cj]):
                            visited[ci, cj] = True
                            cluster.append((ci, cj))
                            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                stack.append((ci + di, cj + dj))

                    # find peak within cluster
                    peak_val = -1
                    peak_i, peak_j = cluster[0]
                    for ci, cj in cluster:
                        if rd_mag[ci, cj] > peak_val:
                            peak_val = rd_mag[ci, cj]
                            peak_i, peak_j = ci, cj

                    detections.append({
                        "range": float(range_axis[peak_i]),
                        "velocity": float(velocity_axis[peak_j]),
                        "power_dB": float(20 * np.log10(peak_val + 1e-30)),
                        "range_bin": peak_i,
                        "doppler_bin": peak_j,
                    })

        return detections

    def process(self, rx_signal: np.ndarray, tx_ref: np.ndarray,
                wavelength: float) -> Dict[str, Any]:
        """Run the full processing pipeline.

        Returns a dict with all intermediate and final products.
        """
        # 1. Matched filter (pulse compression)
        compressed = self.matched_filter(rx_signal, tx_ref)

        # 2. MTI (if enabled)
        if self.cfg.mti_enabled:
            mti_output = self.mti_filter(compressed)
        else:
            mti_output = compressed

        # 3. Range-Doppler map
        rd_map = self.range_doppler_map(mti_output)
        rd_mag = np.abs(rd_map)
        rd_dB = 20 * np.log10(rd_mag + 1e-30)
        rd_dB_norm = rd_dB - rd_dB.max()

        # 4. Axes
        N_range = rd_map.shape[0]
        N_doppler = rd_map.shape[1]

        range_res = self.wf.range_resolution(wavelength)
        range_axis = np.arange(N_range) * range_res

        max_vel = self.wf.max_unambiguous_velocity(wavelength)
        velocity_axis = np.linspace(-max_vel, max_vel, N_doppler, endpoint=False)

        # 5. CFAR detection
        cfar_result = self.cfar_detection(rd_mag)

        # 6. Target estimation
        estimated = self.estimate_targets(rd_map, cfar_result["detection_mask"],
                                          range_axis, velocity_axis)

        return {
            "compressed": compressed,
            "mti_output": mti_output,
            "rd_map": rd_map,
            "rd_dB": rd_dB,
            "rd_dB_norm": rd_dB_norm,
            "range_axis": range_axis,
            "velocity_axis": velocity_axis,
            "detection_mask": cfar_result["detection_mask"],
            "threshold_map": cfar_result["threshold_map"],
            "estimated_targets": estimated,
        }


def plot_results(results: Dict[str, Any],
                 geometry: List[TargetGeometry] = None,
                 save_path: str = "results.png",
                 export_path: str = None):
    """Plot Range-Doppler map, CFAR detections, and target comparison."""

    rd_dB_norm = results["rd_dB_norm"]
    range_axis = results["range_axis"]
    velocity_axis = results["velocity_axis"]
    detection_mask = results["detection_mask"]
    estimated = results["estimated_targets"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Subplot 1: Range-Doppler map ─────────────────────────────────
    ax = axes[0]
    im = ax.pcolormesh(velocity_axis, range_axis, rd_dB_norm,
                       shading="auto", cmap="jet", vmin=-40, vmax=0)
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Range [m]")
    ax.set_title("Range-Doppler Map")
    plt.colorbar(im, ax=ax, label="Power [dB]")

    if geometry:
        for g in geometry:
            ax.plot(g.radial_velocity, g.range_m, "w+",
                    markersize=12, markeredgewidth=2)

    # ── Subplot 2: CFAR detection overlay ────────────────────────────
    ax = axes[1]
    ax.pcolormesh(velocity_axis, range_axis, rd_dB_norm,
                  shading="auto", cmap="jet", vmin=-40, vmax=0, alpha=0.5)
    det_y, det_x = np.where(detection_mask)
    if len(det_y) > 0:
        ax.scatter(velocity_axis[det_x], range_axis[det_y],
                   c="lime", s=3, marker="s", label="CFAR detections")
    if geometry:
        for g in geometry:
            ax.plot(g.radial_velocity, g.range_m, "w+",
                    markersize=12, markeredgewidth=2)
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Range [m]")
    ax.set_title("CFAR Detections")
    ax.legend(fontsize=8)

    # ── Subplot 3: Estimated vs true targets ─────────────────────────
    ax = axes[2]
    ax.axis("off")

    col_labels = ["", "Range [m]", "Vel [m/s]", "Power [dB]"]
    rows = []
    if geometry:
        for g in geometry:
            rows.append([f"True T{g.target_index}",
                         f"{g.range_m:.0f}", f"{g.radial_velocity:.1f}", "-"])
    for i, est in enumerate(estimated):
        rows.append([f"Est  D{i}",
                     f"{est['range']:.0f}",
                     f"{est['velocity']:.1f}",
                     f"{est['power_dB']:.1f}"])

    if rows:
        table = ax.table(cellText=rows, colLabels=col_labels,
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
    ax.set_title("Target Comparison", pad=20)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Results plot saved to {save_path}")

    # optional data export
    if export_path:
        export_data = {
            "rd_map_real": results["rd_map"].real,
            "rd_map_imag": results["rd_map"].imag,
            "rd_dB": results["rd_dB"],
            "range_axis": range_axis,
            "velocity_axis": velocity_axis,
            "detection_mask": detection_mask,
        }
        # add estimated targets as arrays
        if estimated:
            export_data["est_ranges"] = np.array([e["range"] for e in estimated])
            export_data["est_velocities"] = np.array([e["velocity"] for e in estimated])
            export_data["est_powers_dB"] = np.array([e["power_dB"] for e in estimated])
        np.savez_compressed(export_path, **export_data)
        print(f"Data exported to {export_path}")

    return fig, axes
