import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from scenario import Radar, TargetGeometry, C


@dataclass
class WaveformConfig:
    waveform_type: str = "lfm"   # "lfm" or "unmodulated"
    bandwidth: float = 5e6       # chirp bandwidth [Hz] (LFM only)
    pulse_duration: float = 20e-6  # pulse / chirp duration [s]
    PRF: float = 1e3             # pulse repetition frequency [Hz]
    N_pulses: int = 128          # pulses per CPI
    N_samples: int = 512         # fast-time samples per pulse

    @property
    def PRI(self) -> float:
        return 1.0 / self.PRF

    @property
    def chirp_rate(self) -> float:
        return self.bandwidth / self.pulse_duration if self.waveform_type == "lfm" else 0.0

    @property
    def sampling_rate(self) -> float:
        # For stretch/dechirp processing the digitiser samples the full
        # chirp duration, so fs = N_samples / T_chirp.  This ensures each
        # FFT bin maps to exactly one range-resolution cell.
        return self.N_samples / self.pulse_duration

    def range_resolution(self, wavelength: float) -> float:
        if self.waveform_type == "lfm":
            return C / (2 * self.bandwidth)
        else:
            return C * self.pulse_duration / 2

    def max_unambiguous_range(self, wavelength: float) -> float:
        # Only the first N/2 FFT bins represent positive ranges
        return (self.N_samples // 2) * self.range_resolution(wavelength)

    def velocity_resolution(self, wavelength: float) -> float:
        return wavelength / (2 * self.N_pulses * self.PRI)

    def max_unambiguous_velocity(self, wavelength: float) -> float:
        return wavelength / (4 * self.PRI)

    def summary(self, wavelength: float):
        print(f"  Waveform type      : {self.waveform_type.upper()}")
        print(f"  Bandwidth          : {self.bandwidth/1e6:.1f} MHz")
        print(f"  Pulse duration     : {self.pulse_duration*1e6:.1f} µs")
        print(f"  PRF                : {self.PRF:.0f} Hz")
        print(f"  N_pulses           : {self.N_pulses}")
        print(f"  N_samples          : {self.N_samples}")
        print(f"  Range resolution   : {self.range_resolution(wavelength):.1f} m")
        print(f"  Max unamb. range   : {self.max_unambiguous_range(wavelength):.0f} m")
        print(f"  Velocity resolution: {self.velocity_resolution(wavelength):.2f} m/s")
        print(f"  Max unamb. velocity: {self.max_unambiguous_velocity(wavelength):.1f} m/s")


@dataclass
class NoiseConfig:
    thermal_noise_power: float = 0.05     # linear noise power
    clutter_enabled: bool = False
    clutter_cnr_dB: float = 20.0          # clutter-to-noise ratio [dB]
    clutter_correlation: float = 0.95     # pulse-to-pulse correlation (0-1)
    clutter_profile: str = "range_dependent"  # "uniform" or "range_dependent"


class SignalGenerator:
    def __init__(self, radar: Radar, waveform: WaveformConfig,
                 noise: NoiseConfig, geometry: List[TargetGeometry]):
        self.radar = radar
        self.wf = waveform
        self.noise = noise
        self.geometry = geometry

        self.wavelength = radar.wavelength
        self.t_fast = np.arange(self.wf.N_samples) / self.wf.sampling_rate

    def generate_tx_waveform(self) -> np.ndarray:
        """Generate the transmit reference waveform (fast-time vector)."""
        if self.wf.waveform_type == "lfm":
            return np.exp(1j * np.pi * self.wf.chirp_rate * self.t_fast**2)
        else:
            # Unmodulated pulse: constant-frequency carrier over pulse duration
            return np.ones(self.wf.N_samples, dtype=complex)

    def _target_amplitude(self, geom: TargetGeometry) -> float:
        """Compute received voltage amplitude for a target via radar equation.

        Uses the radar equation to compute received SNR relative to thermal
        noise (kTBF), then scales the amplitude so that it is consistent with
        the simulation's thermal_noise_power level.
        """
        Pt = self.radar.tx_power_watts
        G = 10 ** (geom.antenna_gain_dB / 10)  # gain at target direction
        lam = self.wavelength
        sigma = geom.rcs_linear
        R = max(geom.range_m, 1.0)

        # received power from the radar equation [W]
        Pr = (Pt * G**2 * lam**2 * sigma) / ((4 * np.pi)**3 * R**4)

        # physical receiver noise power: kTBF
        k_boltz = 1.38e-23      # Boltzmann constant [J/K]
        T0 = 290.0              # standard temperature [K]
        B = self.wf.bandwidth if self.wf.bandwidth > 0 else 1e6
        noise_figure = 10 ** (3.0 / 10)  # 3 dB noise figure
        Pn_phys = k_boltz * T0 * B * noise_figure

        # single-pulse SNR
        snr = Pr / Pn_phys if Pn_phys > 0 else 0.0

        # amplitude relative to the simulation noise level
        return np.sqrt(max(snr * self.noise.thermal_noise_power, 0.0))

    def _micro_doppler_phase(self, target_type: str, m: int) -> float:
        """Return the micro-Doppler phase offset for pulse m based on target type.

        Models class-specific rotating / oscillating parts as sinusoidal phase
        modulations in slow-time, which produce Doppler sidebands (HERM / blade
        flash lines) in the Range-Doppler map.
        """
        PRI = self.wf.PRI
        t_m = m * PRI  # slow-time at pulse m

        if target_type == "drone":
            # 4 rotors at ~200 Hz blade flash; slight frequency spread across rotors
            rotor_freqs = [195.0, 200.0, 205.0, 210.0]
            amplitude = 0.8  # large sidebands, visible above noise
            phase = sum(amplitude * np.sin(2 * np.pi * f * t_m) for f in rotor_freqs)

        elif target_type == "helicopter":
            # main rotor ~10 Hz (large amplitude) + tail rotor ~60 Hz
            phase = (1.2 * np.sin(2 * np.pi * 10.0 * t_m)
                     + 0.4 * np.sin(2 * np.pi * 60.0 * t_m))

        else:  # fixed_wing / missile
            # turbine blade at ~500 Hz, very small — almost no visible sidebands
            phase = 0.05 * np.sin(2 * np.pi * 500.0 * t_m)

        return phase

    def generate_rx_signal(self) -> np.ndarray:
        """Build the received signal matrix (N_samples x N_pulses) from all targets."""
        N = self.wf.N_samples
        M = self.wf.N_pulses
        rx = np.zeros((N, M), dtype=complex)

        for geom in self.geometry:
            amp = self._target_amplitude(geom)
            R0 = geom.range_m
            vr = geom.radial_velocity

            for m in range(M):
                R_m = R0 + vr * m * self.wf.PRI
                tau = 2 * R_m / C

                if self.wf.waveform_type == "lfm":
                    f_beat = self.wf.chirp_rate * tau
                    phase_fast = 2 * np.pi * f_beat * self.t_fast
                else:
                    # unmodulated: target appears as delayed pulse with phase shift
                    sample_delay = int(round(tau * self.wf.sampling_rate))
                    phase_fast = np.zeros(N)
                    if 0 <= sample_delay < N:
                        phase_fast = np.zeros(N)
                        # model as a phase ramp proportional to carrier delay
                        phase_fast = 2 * np.pi * self.radar.fc * tau * np.ones(N)

                doppler_phase = 2 * np.pi * (2 * vr / self.wavelength) * m * self.wf.PRI
                micro_doppler_phase = self._micro_doppler_phase(geom.target_type, m)

                rx[:, m] += amp * np.exp(1j * (phase_fast + doppler_phase
                                                + micro_doppler_phase))

        return rx

    def add_noise(self, rx_signal: np.ndarray) -> np.ndarray:
        """Add thermal white Gaussian noise."""
        sigma = np.sqrt(self.noise.thermal_noise_power / 2)
        noise = sigma * (np.random.randn(*rx_signal.shape)
                         + 1j * np.random.randn(*rx_signal.shape))
        return rx_signal + noise

    def add_clutter(self, rx_signal: np.ndarray) -> np.ndarray:
        """Add clutter to the received signal.

        Clutter is modelled as correlated complex noise across slow-time,
        with power that is either uniform or range-dependent (scales as 1/R²).
        """
        N = self.wf.N_samples
        M = self.wf.N_pulses

        clutter_noise_linear = self.noise.thermal_noise_power * 10 ** (self.noise.clutter_cnr_dB / 10)

        # range-bin power profile
        if self.noise.clutter_profile == "range_dependent":
            range_bins = np.arange(1, N + 1, dtype=float)
            # power ∝ 1/R², normalise so mean equals clutter_noise_linear
            profile = 1.0 / range_bins**2
            profile *= clutter_noise_linear / profile.mean()
        else:
            profile = clutter_noise_linear * np.ones(N)

        # generate correlated clutter across slow-time
        rho = self.noise.clutter_correlation
        clutter = np.zeros((N, M), dtype=complex)

        # first pulse: uncorrelated sample
        clutter[:, 0] = (np.sqrt(profile / 2)
                         * (np.random.randn(N) + 1j * np.random.randn(N)))

        # subsequent pulses: AR(1) correlation model
        for m in range(1, M):
            innovation = (np.sqrt(profile * (1 - rho**2) / 2)
                          * (np.random.randn(N) + 1j * np.random.randn(N)))
            clutter[:, m] = rho * clutter[:, m - 1] + innovation

        return rx_signal + clutter

    def get_signal(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run the full signal generation pipeline.

        Returns
        -------
        rx_signal : np.ndarray, shape (N_samples, N_pulses)
            The complete received signal (targets + noise + clutter).
        tx_ref : np.ndarray, shape (N_samples,)
            The transmit reference waveform.
        """
        tx_ref = self.generate_tx_waveform()
        rx = self.generate_rx_signal()
        rx = self.add_noise(rx)
        if self.noise.clutter_enabled:
            rx = self.add_clutter(rx)
        return rx, tx_ref
