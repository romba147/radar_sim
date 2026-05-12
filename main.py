import numpy as np

from scenario import Radar, Target, Scenario
from signal_gen import WaveformConfig, NoiseConfig, SignalGenerator
from processing import ProcessingConfig, WindowType, RadarProcessor, plot_results
from fusion import associate_and_fuse, print_fusion_report
from interceptor import InterceptorSystem, InterceptBlackbox, \
                        print_intercept_table, plot_intercept_matrix


def main():
    # ═══════════════════════════════════════════════════════════════════
    # PART 1 — Geometric Scenario (multiple radars)
    # ═══════════════════════════════════════════════════════════════════

    radars = [
        Radar(
            position=[0, 0, 0],
            fc=10e9,                  # 10 GHz (X-band)
            tx_power_dBm=60,          # 1 kW
            antenna_gain_dB=30,
            antenna_beamwidth=5.0,
            look_azimuth=10.0,
            look_elevation=0.0,
        ),
        Radar(
            position=[0, 3000, 0],
            fc=9e9,                   # 9 GHz (X-band)
            tx_power_dBm=58,          # ~630 W
            antenna_gain_dB=30,
            antenna_beamwidth=8.0,    # wider beam
            look_azimuth=-25.0,      # looking toward targets
            look_elevation=0.0,
        ),
        Radar(
            position=[-2000, 1500, 0],
            fc=10e9,                  # 10 GHz
            tx_power_dBm=60,          # 1 kW
            antenna_gain_dB=30,
            antenna_beamwidth=8.0,    # wider beam
            look_azimuth=0.0,        # looking toward targets
            look_elevation=0.0,
        ),
    ]

    targets = [
        Target(position=[5000, 800, 0],   velocity=[25, 5, 0],   rcs_dbsm=10),
        Target(position=[3000, 200, 50],  velocity=[-15, 0, 0],  rcs_dbsm=5),
        Target(position=[7000, 1500, 0],  velocity=[20, -10, 0], rcs_dbsm=15),
        Target(position=[2000, 100, 0],   velocity=[10, 2, 0],   rcs_dbsm=0),
        Target(position=[6000, -500, 0],  velocity=[-30, 8, 0],  rcs_dbsm=8),
    ]

    scenario = Scenario(radars, targets)
    geometry = scenario.compute_geometry()

    print("\n" + "=" * 62)
    print("              PART 1 -- GEOMETRIC SCENARIO")
    print("=" * 62)
    scenario.summary(geometry)
    # scenario plot is generated after interceptor systems are defined (see below)

    # ═══════════════════════════════════════════════════════════════════
    # PART 2 & 3 — Per-Radar Signal Generation & Processing
    # ═══════════════════════════════════════════════════════════════════

    # Each radar has its own waveform, noise, and processing config
    radar_configs = [
        {
            "waveform": WaveformConfig(
                waveform_type="lfm",
                bandwidth=10e6,
                pulse_duration=10e-6,
                PRF=5000,
                N_pulses=256,
                N_samples=1024,
            ),
            "noise": NoiseConfig(
                thermal_noise_power=0.01,
                clutter_enabled=True,
                clutter_cnr_dB=15,
                clutter_correlation=0.98,
                clutter_profile="range_dependent",
            ),
            "processing": ProcessingConfig(
                range_window=WindowType.HAMMING,
                doppler_window=WindowType.HANNING,
                cfar_guard_cells=4,
                cfar_training_cells=16,
                cfar_threshold_factor=20.0,
                mti_enabled=True,
                mti_order=1,
            ),
        },
        {
            "waveform": WaveformConfig(
                waveform_type="lfm",
                bandwidth=8e6,
                pulse_duration=12e-6,
                PRF=4000,
                N_pulses=128,
                N_samples=512,
            ),
            "noise": NoiseConfig(
                thermal_noise_power=0.015,
                clutter_enabled=True,
                clutter_cnr_dB=12,
                clutter_correlation=0.95,
                clutter_profile="range_dependent",
            ),
            "processing": ProcessingConfig(
                range_window=WindowType.HANNING,
                doppler_window=WindowType.HAMMING,
                cfar_guard_cells=3,
                cfar_training_cells=12,
                cfar_threshold_factor=18.0,
                mti_enabled=True,
                mti_order=1,
            ),
        },
        {
            "waveform": WaveformConfig(
                waveform_type="lfm",
                bandwidth=12e6,
                pulse_duration=8e-6,
                PRF=6000,
                N_pulses=256,
                N_samples=1024,
            ),
            "noise": NoiseConfig(
                thermal_noise_power=0.008,
                clutter_enabled=True,
                clutter_cnr_dB=18,
                clutter_correlation=0.97,
                clutter_profile="range_dependent",
            ),
            "processing": ProcessingConfig(
                range_window=WindowType.HAMMING,
                doppler_window=WindowType.HANNING,
                cfar_guard_cells=4,
                cfar_training_cells=16,
                cfar_threshold_factor=22.0,
                mti_enabled=True,
                mti_order=1,
            ),
        },
    ]

    per_radar_detections = {}

    for r_idx, radar in enumerate(radars):
        cfg = radar_configs[r_idx]
        waveform = cfg["waveform"]
        noise = cfg["noise"]
        proc_config = cfg["processing"]

        print(f"\n{'=' * 62}")
        print(f"      RADAR {r_idx} — SIGNAL GEN & PROCESSING")
        print(f"{'=' * 62}")
        print(f"  Position: {radar.position}, fc={radar.fc/1e9:.2f} GHz")
        print(f"  Waveform: {waveform.waveform_type.upper()}, "
              f"BW={waveform.bandwidth/1e6:.1f} MHz, "
              f"PRF={waveform.PRF:.0f} Hz, "
              f"N_pulses={waveform.N_pulses}")

        # signal generation
        sig_gen = SignalGenerator(radar, waveform, noise, geometry[r_idx])
        rx_signal, tx_ref = sig_gen.get_signal()

        print(f"  Rx signal shape: {rx_signal.shape}")

        # processing
        processor = RadarProcessor(waveform, proc_config)
        results = processor.process(rx_signal, tx_ref, radar.wavelength)

        estimated = results["estimated_targets"]
        per_radar_detections[r_idx] = estimated

        print(f"  CFAR detections: {len(estimated)}")
        for d_idx, det in enumerate(estimated):
            print(f"    D{d_idx}: range={det['range']:.0f}m, "
                  f"velocity={det['velocity']:.1f}m/s, "
                  f"power={det['power_dB']:.1f}dB")

        # per-radar results plot
        plot_results(results, geometry[r_idx],
                     save_path=f"results_radar{r_idx}.png",
                     export_path=f"results_radar{r_idx}.npz")

    # ═══════════════════════════════════════════════════════════════════
    # PART 3.5 — Multi-Radar Fusion
    # ═══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 62)
    print("           PART 3.5 -- MULTI-RADAR FUSION")
    print("=" * 62)

    fused_targets = associate_and_fuse(radars, per_radar_detections)
    print_fusion_report(fused_targets)

    # ═══════════════════════════════════════════════════════════════════
    # PART 4 — Intercept Probability (position-based)
    # ═══════════════════════════════════════════════════════════════════

    systems = [
        InterceptorSystem(
            name="Short-Range SAM",
            position=[1000, 500, 0],
            min_range=500,
            max_range=4000,
            max_target_velocity=20.0,
            reaction_time=5.0,
            salvo_size=2,
        ),
        InterceptorSystem(
            name="Medium-Range SAM",
            position=[0, 1500, 0],
            min_range=1000,
            max_range=7000,
            max_target_velocity=35.0,
            reaction_time=8.0,
            salvo_size=1,
        ),
        InterceptorSystem(
            name="Long-Range SAM",
            position=[-1000, 2000, 0],
            min_range=3000,
            max_range=10000,
            max_target_velocity=50.0,
            reaction_time=15.0,
            salvo_size=1,
        ),
        InterceptorSystem(
            name="CIWS (Gun)",
            position=[500, 0, 0],
            min_range=100,
            max_range=2000,
            max_target_velocity=40.0,
            reaction_time=2.0,
            salvo_size=4,
        ),
    ]

    print("\n" + "=" * 62)
    print("              PART 4 -- INTERCEPT ASSESSMENT")
    print("=" * 62)
    print(f"  {len(systems)} interceptor systems vs "
          f"{len(fused_targets)} fused targets\n")

    blackbox = InterceptBlackbox(systems)
    P = blackbox.evaluate(fused_targets)

    print("=== Intercept Probability Matrix [%] ===\n")
    print_intercept_table(systems, fused_targets, P)

    plot_intercept_matrix(systems, fused_targets, P,
                          save_path="intercept_matrix.png")

    # ── Scenario plot (with interceptor positions) ───────────────────
    scenario.plot_scenario(geometry, interceptor_systems=systems,
                           save_path="scenario_plot.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
