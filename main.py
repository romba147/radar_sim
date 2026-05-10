import numpy as np

from scenario import Radar, Target, Scenario
from signal_gen import WaveformConfig, NoiseConfig, SignalGenerator
from processing import ProcessingConfig, WindowType, RadarProcessor, plot_results
from interceptor import InterceptorSystem, InterceptBlackbox, \
                        print_intercept_table, plot_intercept_matrix


def main():
    # ═══════════════════════════════════════════════════════════════════
    # PART 1 — Geometric Scenario
    # ═══════════════════════════════════════════════════════════════════

    radar = Radar(
        position=[0, 0, 0],
        fc=10e9,                  # 10 GHz (X-band)
        tx_power_dBm=60,          # 1 kW
        antenna_gain_dB=30,
        antenna_beamwidth=5.0,    # 5° beamwidth
        look_azimuth=10.0,        # boresight at 10° azimuth
        look_elevation=0.0,
    )

    targets = [
        Target(position=[5000, 800, 0],   velocity=[25, 5, 0],   rcs_dbsm=10),
        Target(position=[3000, 200, 50],  velocity=[-15, 0, 0],  rcs_dbsm=5),
        Target(position=[7000, 1500, 0],  velocity=[20, -10, 0], rcs_dbsm=15),
        Target(position=[2000, 100, 0],   velocity=[10, 2, 0],   rcs_dbsm=0),
        Target(position=[6000, -500, 0],  velocity=[-30, 8, 0],  rcs_dbsm=8),
    ]

    scenario = Scenario(radar, targets)
    geometry = scenario.compute_geometry()

    print("\n" + "=" * 62)
    print("              PART 1 -- GEOMETRIC SCENARIO")
    print("=" * 62)
    scenario.summary(geometry)
    scenario.plot_scenario(geometry, save_path="scenario_plot.png")

    # ═══════════════════════════════════════════════════════════════════
    # PART 2 — Signal Generation
    # ═══════════════════════════════════════════════════════════════════

    waveform = WaveformConfig(
        waveform_type="lfm",
        bandwidth=10e6,           # 10 MHz → 15 m range resolution
        pulse_duration=10e-6,     # 10 µs chirp
        PRF=5000,                 # 5 kHz PRF
        N_pulses=256,
        N_samples=1024,
    )

    noise = NoiseConfig(
        thermal_noise_power=0.01,
        clutter_enabled=True,
        clutter_cnr_dB=15,
        clutter_correlation=0.98,
        clutter_profile="range_dependent",
    )

    print("\n" + "=" * 62)
    print("              PART 2 -- SIGNAL GENERATION")
    print("=" * 62)
    print("=== Waveform Configuration ===")
    waveform.summary(radar.wavelength)
    print(f"\n=== Noise Configuration ===")
    print(f"  Thermal noise power : {noise.thermal_noise_power}")
    print(f"  Clutter enabled     : {noise.clutter_enabled}")
    print(f"  Clutter CNR         : {noise.clutter_cnr_dB} dB")
    print(f"  Clutter correlation : {noise.clutter_correlation}")
    print(f"  Clutter profile     : {noise.clutter_profile}")

    sig_gen = SignalGenerator(radar, waveform, noise, geometry)
    rx_signal, tx_ref = sig_gen.get_signal()

    print(f"\n  Rx signal shape     : {rx_signal.shape}")
    print(f"  Tx reference length : {tx_ref.shape[0]}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 3 — Signal Processing
    # ═══════════════════════════════════════════════════════════════════

    proc_config = ProcessingConfig(
        range_window=WindowType.HAMMING,
        doppler_window=WindowType.HANNING,
        cfar_guard_cells=4,
        cfar_training_cells=16,
        cfar_threshold_factor=20.0,
        mti_enabled=True,
        mti_order=1,
    )

    print("\n" + "=" * 62)
    print("              PART 3 -- SIGNAL PROCESSING")
    print("=" * 62)
    print("=== Processing Configuration ===")
    print(f"  Range window        : {proc_config.range_window.value}")
    print(f"  Doppler window      : {proc_config.doppler_window.value}")
    print(f"  MTI enabled         : {proc_config.mti_enabled} (order {proc_config.mti_order})")
    print(f"  CFAR guard cells    : {proc_config.cfar_guard_cells}")
    print(f"  CFAR training cells : {proc_config.cfar_training_cells}")
    print(f"  CFAR threshold      : {proc_config.cfar_threshold_factor}")

    processor = RadarProcessor(waveform, proc_config)
    results = processor.process(rx_signal, tx_ref, radar.wavelength)

    # ═══════════════════════════════════════════════════════════════════
    # Results
    # ═══════════════════════════════════════════════════════════════════

    estimated = results["estimated_targets"]
    print(f"\n=== CFAR Detections: {len(estimated)} targets ===")
    for i, det in enumerate(estimated):
        print(f"  Detection {i}: range={det['range']:.0f} m, "
              f"velocity={det['velocity']:.1f} m/s, "
              f"power={det['power_dB']:.1f} dB")

    print("\n=== True vs Estimated Comparison ===")
    print(f"  {'True':>6s}  {'R_true':>8s}  {'V_true':>8s}  |  "
          f"{'Est':>6s}  {'R_est':>8s}  {'V_est':>8s}  {'dR':>6s}  {'dV':>6s}")
    print("  " + "-" * 70)
    for g in geometry:
        # find closest estimated detection
        best = None
        best_dist = float("inf")
        for est in estimated:
            dist = abs(est["range"] - g.range_m) + abs(est["velocity"] - g.radial_velocity)
            if dist < best_dist:
                best_dist = dist
                best = est
        if best:
            dr = abs(best["range"] - g.range_m)
            dv = abs(best["velocity"] - g.radial_velocity)
            print(f"  T{g.target_index:4d}  {g.range_m:8.0f}  {g.radial_velocity:8.1f}  |  "
                  f"{'match':>6s}  {best['range']:8.0f}  {best['velocity']:8.1f}  "
                  f"{dr:6.0f}  {dv:6.1f}")
        else:
            print(f"  T{g.target_index:4d}  {g.range_m:8.0f}  {g.radial_velocity:8.1f}  |  "
                  f"{'MISS':>6s}")

    # ── Plot & export ────────────────────────────────────────────────
    plot_results(results, geometry,
                 save_path="results.png",
                 export_path="results.npz")

    # ═══════════════════════════════════════════════════════════════════
    # PART 4 — Intercept Probability Blackbox
    # ═══════════════════════════════════════════════════════════════════

    systems = [
        InterceptorSystem(
            name="Short-Range SAM",
            min_range=500,
            max_range=4000,
            max_target_velocity=20.0,
            reaction_time=5.0,
            salvo_size=2,
        ),
        InterceptorSystem(
            name="Medium-Range SAM",
            min_range=1000,
            max_range=7000,
            max_target_velocity=35.0,
            reaction_time=8.0,
            salvo_size=1,
        ),
        InterceptorSystem(
            name="Long-Range SAM",
            min_range=3000,
            max_range=10000,
            max_target_velocity=50.0,
            reaction_time=15.0,
            salvo_size=1,
        ),
        InterceptorSystem(
            name="CIWS (Gun)",
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
    print(f"  {len(systems)} interceptor systems vs {len(estimated)} radar-estimated targets\n")

    blackbox = InterceptBlackbox(systems)
    P = blackbox.evaluate(estimated)

    print("=== Intercept Probability Matrix [%] ===\n")
    print_intercept_table(systems, estimated, P)

    plot_intercept_matrix(systems, estimated, P,
                          save_path="intercept_matrix.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
