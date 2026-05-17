import os
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

    # ── 4a: Train XGBoost model if not already trained ───────────────
    model_path = "xgb_intercept_model.json"
    data_path = "mc_engagement_data.npz"

    if not os.path.exists(model_path):
        print("\n" + "=" * 62)
        print("       PART 4a -- TRAINING XGBOOST INTERCEPT MODEL")
        print("=" * 62)
        from train_model import train
        train(data_path=data_path, model_path=model_path,
              n_samples=50000, seed=42, verbose=True)

    # ── 4b: Analytical intercept assessment ──────────────────────────
    print("\n" + "=" * 62)
    print("         PART 4b -- ANALYTICAL INTERCEPT ASSESSMENT")
    print("=" * 62)
    print(f"  {len(systems)} interceptor systems vs "
          f"{len(fused_targets)} fused targets\n")

    blackbox_analytical = InterceptBlackbox(systems, use_ml=False)
    P_analytical = blackbox_analytical.evaluate(fused_targets)

    print("=== Analytical Intercept Probability Matrix [%] ===\n")
    print_intercept_table(systems, fused_targets, P_analytical)

    plot_intercept_matrix(systems, fused_targets, P_analytical,
                          save_path="intercept_matrix_analytical.png")

    # ── 4c: ML-based intercept assessment ────────────────────────────
    print("\n" + "=" * 62)
    print("           PART 4c -- ML INTERCEPT ASSESSMENT")
    print("=" * 62)

    blackbox_ml = InterceptBlackbox(systems, use_ml=True,
                                    model_path=model_path)
    P_ml = blackbox_ml.evaluate(fused_targets)

    print("=== ML (XGBoost) Intercept Probability Matrix [%] ===\n")
    print_intercept_table(systems, fused_targets, P_ml)

    plot_intercept_matrix(systems, fused_targets, P_ml,
                          save_path="intercept_matrix_ml.png")

    # ── 4d: Side-by-side comparison ──────────────────────────────────
    print("\n" + "=" * 62)
    print("        PART 4d -- ANALYTICAL vs ML COMPARISON")
    print("=" * 62)

    _print_comparison(systems, fused_targets, P_analytical, P_ml)
    _plot_comparison(systems, fused_targets, P_analytical, P_ml)

    # ── Scenario plot (with interceptor positions) ───────────────────
    scenario.plot_scenario(geometry, interceptor_systems=systems,
                           save_path="scenario_plot.png")

    print("\nDone.")


def _print_comparison(systems, fused_targets, P_ana, P_ml):
    """Print side-by-side analytical vs ML probabilities."""
    n_sys, n_tgt = P_ana.shape
    print(f"\n  {'System':<20s}", end="")
    for j in range(n_tgt):
        print(f"  FT{j} Ana/ML  ", end="")
    print()
    print("  " + "-" * (20 + n_tgt * 16))
    for i, sys in enumerate(systems):
        print(f"  {sys.name:<20s}", end="")
        for j in range(n_tgt):
            a = P_ana[i, j] * 100
            m = P_ml[i, j] * 100
            print(f"  {a:5.1f}/{m:5.1f}%  ", end="")
        print()

    # Aggregate metrics
    diff = np.abs(P_ana - P_ml) * 100
    print(f"\n  Mean absolute difference: {diff.mean():.1f}%")
    print(f"  Max absolute difference:  {diff.max():.1f}%")
    corr = np.corrcoef(P_ana.ravel(), P_ml.ravel())[0, 1] if P_ana.size > 1 else float('nan')
    print(f"  Correlation:              {corr:.4f}")


def _plot_comparison(systems, fused_targets, P_ana, P_ml):
    """Generate side-by-side heatmaps and scatter comparison plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    n_sys, n_tgt = P_ana.shape

    sys_labels = [s.name for s in systems]
    tgt_labels = [f"FT{j}" for j in range(n_tgt)]

    # Analytical heatmap
    im0 = axes[0].imshow(P_ana * 100, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    axes[0].set_xticks(range(n_tgt)); axes[0].set_xticklabels(tgt_labels, fontsize=8)
    axes[0].set_yticks(range(n_sys)); axes[0].set_yticklabels(sys_labels, fontsize=8)
    axes[0].set_title("Analytical Model")
    for i in range(n_sys):
        for j in range(n_tgt):
            c = "white" if P_ana[i,j]*100 < 30 or P_ana[i,j]*100 > 70 else "black"
            axes[0].text(j, i, f"{P_ana[i,j]*100:.1f}", ha="center", va="center", fontsize=8, color=c)
    plt.colorbar(im0, ax=axes[0], label="%")

    # ML heatmap
    im1 = axes[1].imshow(P_ml * 100, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    axes[1].set_xticks(range(n_tgt)); axes[1].set_xticklabels(tgt_labels, fontsize=8)
    axes[1].set_yticks(range(n_sys)); axes[1].set_yticklabels(sys_labels, fontsize=8)
    axes[1].set_title("XGBoost ML Model")
    for i in range(n_sys):
        for j in range(n_tgt):
            c = "white" if P_ml[i,j]*100 < 30 or P_ml[i,j]*100 > 70 else "black"
            axes[1].text(j, i, f"{P_ml[i,j]*100:.1f}", ha="center", va="center", fontsize=8, color=c)
    plt.colorbar(im1, ax=axes[1], label="%")

    # Scatter: analytical vs ML
    axes[2].scatter(P_ana.ravel() * 100, P_ml.ravel() * 100,
                    c="steelblue", s=80, edgecolors="navy", alpha=0.8)
    axes[2].plot([0, 100], [0, 100], "k--", alpha=0.4, label="y = x")
    axes[2].set_xlabel("Analytical P(intercept) [%]")
    axes[2].set_ylabel("ML P(intercept) [%]")
    axes[2].set_title("Analytical vs ML")
    axes[2].set_xlim(-5, 105); axes[2].set_ylim(-5, 105)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Intercept Probability Comparison: Analytical vs XGBoost", fontsize=13)
    plt.tight_layout()
    fig.savefig("intercept_comparison.png", dpi=150)
    plt.close()
    print(f"  Comparison plot saved → intercept_comparison.png")


if __name__ == "__main__":
    main()
