# GeneralSim — Radar Simulation & Intercept Prediction System

An end-to-end radar simulation pipeline that covers scenario definition, signal generation, signal processing, multi-radar fusion, target classification, and intercept probability assessment using both analytical and ML-based (XGBoost) models.

---

## Project Structure

```
GeneralSim/
├── app.py                   # Interactive Streamlit dashboard (recommended entry point)
├── main.py                  # Headless orchestration entry point
├── scenario.py              # Radar/target geometry definitions
├── signal_gen.py            # Radar signal generation
├── processing.py            # Signal processing & CFAR detection
├── feature_extraction.py    # Feature engineering for ML
├── fusion.py                # Multi-radar detection fusion
├── classifier.py            # Micro-Doppler target classification
├── interceptor.py           # Intercept probability assessment
├── mc_simulator.py          # Monte Carlo engagement simulation
├── train_model.py           # XGBoost model training
├── check_python.py          # Environment diagnostic utility
├── xgb_intercept_model.json # Trained XGBoost model
└── mc_engagement_data.npz   # Monte Carlo training dataset
```

---

## Data Flow

```
scenario.py  →  signal_gen.py  →  processing.py  →  fusion.py
                                                         ↓
                                               classifier.py  (micro-Doppler)
                                                         ↓
                                               feature_extraction.py
                                                         ↓
                                               interceptor.py  ←  train_model.py
                                                                        ↑
                                                               mc_simulator.py
```

---

## File & Function Reference

### `main.py` — Orchestration Script

Entry point that runs the full simulation pipeline.

| Function | Description |
|---|---|
| `main()` | Orchestrates the complete workflow: defines scenario, runs per-radar signal generation and processing, performs multi-radar fusion, optionally trains the XGBoost model, then runs both analytical and ML intercept assessments for comparison. |
| `_print_comparison(systems, fused_targets, P_ana, P_ml)` | Prints a side-by-side table of analytical vs ML intercept probabilities with aggregate metrics (mean difference, max difference, correlation). |
| `_plot_comparison(systems, fused_targets, P_ana, P_ml)` | Generates a three-panel figure: analytical heatmap, ML heatmap, and scatter plot comparing the two probability estimates. |

---

### `scenario.py` — Geometric Scenario Definition

Defines radar and target objects and computes their geometric relationships.

#### Classes

**`Radar`** — Represents a radar system.

| Method / Property | Description |
|---|---|
| `beam_pattern(azimuth_deg, elevation_deg)` | Computes antenna gain (dB) at a given direction using a sinc² pattern. |
| `wavelength` | Returns wavelength derived from carrier frequency. |
| `tx_power_watts` | Converts transmit power from dBm to watts. |
| `antenna_gain_linear` | Converts antenna gain from dB to linear scale. |

**`Target`** — Represents a moving target. Fields: `position`, `velocity`, `rcs_dbsm`, `target_type` (`"drone"`, `"helicopter"`, or `"fixed_wing"`, default `"fixed_wing"`).

| Property | Description |
|---|---|
| `rcs_linear` | Converts RCS from dBsm to linear scale. |

**`TargetGeometry`** — Stores computed geometric relationship between one radar and one target (range, radial velocity, azimuth, elevation, antenna gain, RCS).

**`Scenario`** — Manages collections of radars and targets.

| Method | Description |
|---|---|
| `compute_geometry()` | Computes range, radial velocity, azimuth, elevation, and antenna gain for every radar–target pair. |
| `summary(geometry)` | Prints a formatted table of all geometric parameters. |
| `plot_scenario(geometry, interceptor_systems, save_path)` | Generates a 2D top-down visualization showing radars with beam sectors, targets with velocity vectors, and interceptor systems with engagement range rings. |

---

### `signal_gen.py` — Signal Generation

Generates realistic received radar signals with target returns, thermal noise, and clutter.

#### Classes

**`WaveformConfig`** — Defines waveform parameters (type, bandwidth, pulse duration, PRF, number of pulses/samples).

| Method | Description |
|---|---|
| `range_resolution(wavelength)` | Computes range resolution from bandwidth. |
| `max_unambiguous_range(wavelength)` | Returns the maximum unambiguous range. |
| `velocity_resolution(wavelength)` | Computes velocity resolution from Doppler processing. |
| `max_unambiguous_velocity(wavelength)` | Returns the maximum unambiguous radial velocity. |
| `summary(wavelength)` | Prints a summary of waveform specifications. |

**`NoiseConfig`** — Defines noise and clutter parameters (thermal noise power, clutter enable, CNR, clutter correlation, profile type).

**`SignalGenerator`** — Generates the full received signal.

| Method | Description |
|---|---|
| `generate_tx_waveform()` | Creates an LFM chirp or unmodulated pulse reference waveform. |
| `_target_amplitude(geom)` | Computes received signal amplitude for a single target using the radar range equation. |
| `generate_rx_signal()` | Builds the (N_samples × N_pulses) received signal matrix from all targets, applying range delay and Doppler shift. |
| `add_noise(rx_signal)` | Adds complex white Gaussian thermal noise. |
| `add_clutter(rx_signal)` | Adds correlated clutter using an AR(1) model with a range-dependent profile. |
| `get_signal()` | Full pipeline: generates TX waveform then returns the composite RX signal (targets + noise + optional clutter). |

---

### `processing.py` — Radar Signal Processing

Processes raw signals to produce range-Doppler maps and detect targets via CFAR.

#### Classes / Enums

**`WindowType`** — Enum of supported window functions: `RECTANGULAR`, `HANNING`, `HAMMING`, `BLACKMAN`, `KAISER`.

**`ProcessingConfig`** — Holds processing parameters: range/Doppler windows, CFAR guard cells, training cells, threshold factor, and MTI order.

**`RadarProcessor`** — Performs the processing chain.

| Method | Description |
|---|---|
| `matched_filter(rx_signal, tx_ref)` | Applies pulse compression via FFT-based matched filtering across all pulses. |
| `mti_filter(rx_signal)` | Performs clutter cancellation via pulse-to-pulse subtraction (1st or 2nd order). |
| `range_doppler_map(rx_signal)` | Computes a windowed 2D FFT to produce the range-Doppler map. |
| `cfar_detection(rd_map_mag)` | Runs a 2D Cell-Averaging CFAR detector over the magnitude range-Doppler map. |
| `estimate_targets(rd_map, detection_mask, range_axis, velocity_axis)` | Extracts target estimates from the detection map using flood-fill clustering and peak finding. |
| `process(rx_signal, tx_ref, wavelength)` | Full pipeline returning: compressed signal, MTI output, range-Doppler map, detection mask, and estimated targets. |

#### Module-Level Functions

| Function | Description |
|---|---|
| `_make_window(window_type, length, kaiser_beta)` | Factory that returns window coefficient arrays for a given `WindowType`. |
| `plot_results(results, geometry, save_path, export_path)` | Generates a 3-panel figure (range-Doppler map, CFAR detections overlay, target comparison table) and optionally exports processed data to an NPZ file. |

---

### `feature_extraction.py` — Feature Engineering

Extracts a consistent 17-element feature vector used by both the training pipeline and the ML inference path.

**Module constants:** `FEATURE_NAMES` (list of 17 names), `N_FEATURES = 17`.

| Function | Description |
|---|---|
| `extract_from_system_target(system, fused_target)` | Extracts the 17-element feature vector from an `InterceptorSystem` and a `FusedTarget`. Features cover geometry (distance, closing speed, azimuth, elevation), track quality, radar count, system envelope (max/min range, max velocity, reaction time, salvo size), and derived ratios (range fraction, speed fraction, in-envelope flag). |
| `extract_matrix(systems, fused_targets)` | Builds the full (n_systems × n_targets) × 17 feature matrix. |
| `dataset_to_Xy(data)` | Converts a Monte Carlo dataset dictionary (from NPZ) into a feature matrix `X` and label vector `y`. |

---

### `fusion.py` — Multi-Radar Fusion

Associates detections from multiple radars into fused targets using triangulation and least-squares estimation.

#### Classes

**`FusedTarget`** — Stores a fused estimate: Cartesian position, velocity vector, number of contributing radars, per-radar detection info, best power, track quality, position method, and optional classification result (`target_type`, `classification_confidence`).

#### Functions

| Function | Description |
|---|---|
| `_boresight_direction(radar)` | Returns the unit vector along a radar's boresight. |
| `_approx_cartesian(radar, detection)` | Approximates Cartesian target position as radar position + range × boresight (used as a fallback). |
| `_triangulate_position(radars, detections, radar_indices)` | Least-squares triangulation that solves `‖x − pᵢ‖² = rᵢ²` to find Cartesian position from range measurements across multiple radars. |
| `_estimate_velocity_vector(radars, detections, radar_indices, target_pos)` | Solves for 3D velocity by fitting radial velocity measurements via least-squares. |
| `associate_and_fuse(radars, per_radar_detections, gate_factor)` | Main fusion function. Converts detections to approximate Cartesian positions, runs greedy nearest-neighbour association with spatial gating, triangulates position, estimates velocity, and returns a list of `FusedTarget` objects. |
| `print_fusion_report(fused_targets)` | Prints a formatted table of fused target parameters. |

---

### `classifier.py` — Micro-Doppler Target Classification

Classifies fused targets as `drone`, `helicopter`, or `fixed_wing` by analysing the Doppler spectral profile at a detected target's range bin.

| Function | Description |
|---|---|
| `extract_doppler_features(rd_map, detection, velocity_axis)` | Extracts four micro-Doppler spectral features from the Range-Doppler map at a target's range bin: Doppler bandwidth (10-dB width), sideband symmetry, spectral entropy, and peak sideband offset. Requires `range_bin` and `doppler_bin` from the per-radar detection dict. |
| `classify_target(features)` | Rule-based classifier that maps the four spectral features to a `(label, confidence)` tuple. Drones exhibit wide, symmetric Doppler spread; helicopters show moderate bandwidth with low-frequency sidebands; fixed-wing targets have narrow bulk Doppler with minimal sidebands. |

Classification is run in the simulation loop after multi-radar fusion. Each `FusedTarget` uses the highest-SNR contributing detection to select the Range-Doppler map, and the result is stored in `ft.target_type` and `ft.classification_confidence`. When **Enable Target Classification** is unchecked, the manually-set sidebar type is used instead.

---

### `interceptor.py` — Intercept Probability Assessment

Computes intercept probability for each (interceptor system, target) pair using an analytical or ML model.

#### Classes

**`InterceptorSystem`** — Defines an interceptor system: name, position, max/min engagement range, max target velocity, reaction time, salvo size.

**`InterceptBlackbox`** — Evaluates intercept probabilities.

| Method | Description |
|---|---|
| `evaluate(targets)` | Returns an (N_systems × N_targets) probability matrix. |
| `_compute(system, target)` | Routes to either the analytical or ML calculation. |
| `_compute_analytical(system, target)` | Analytical model: checks range/speed envelope, computes `range_factor` and `speed_factor` as linear falloffs, applies a square-root track-quality scaling, then scales up for a multi-round salvo assuming independent shots. |
| `_compute_ml(system, target)` | Extracts features and queries the loaded XGBoost model to predict intercept probability. |

#### Module-Level Functions

| Function | Description |
|---|---|
| `print_intercept_table(systems, targets, P)` | Prints the probability matrix as a formatted table with per-target and per-system details. |
| `plot_intercept_matrix(systems, targets, P, save_path)` | Generates a heat-map visualization of the probability matrix with annotated cell values. |

---

### `mc_simulator.py` — Monte Carlo Engagement Simulation

Generates labeled training data by simulating stochastic target-intercept engagements.

#### Classes

**`EngagementConfig`** — Tunable simulation parameters: interceptor speed factor, lethal radius, target evasive maneuver settings (g-load, probability, count), guidance noise, scenario randomization bounds, and track quality/radar feature distributions.

#### Functions

| Function | Description |
|---|---|
| `_propagate_target(pos, vel, duration, cfg, rng)` | Propagates a target trajectory with stochastic evasive maneuvers. Schedules random events, applies perpendicular turn accelerations while maintaining speed, and returns an (N_steps+1, 3) trajectory array. |
| `_simulate_intercept(interceptor_pos, interceptor_speed, lethal_radius, max_flight_range, target_trajectory, dt, guidance_noise_std, rng)` | Simulates interceptor flight using proportional navigation with additive noise. Tracks minimum miss distance and stops on intercept, range exceedance, or trajectory end. |
| `_single_engagement(system_params, target_params, cfg, rng)` | Runs one full MC engagement: computes features, simulates target propagation with reaction-time delay, launches a salvo of independent interceptors, and returns a feature dict + binary hit label. |
| `_random_system(rng, reference_systems)` | Generates a randomised interceptor system by perturbing a reference system. |
| `_random_target(rng, sys_pos, cfg)` | Generates a random target relative to an interceptor system position within configured bounds. |
| `generate_dataset(n_samples, reference_systems, cfg, seed, verbose)` | Main function: generates `n_samples` labelled engagement records, each with all 17 features. |
| `save_dataset(data, path)` | Saves a dataset dictionary to a compressed NPZ file. |
| `load_dataset(path)` | Loads a dataset dictionary from an NPZ file. |

---

### `train_model.py` — XGBoost Model Training

Trains an XGBoost binary classifier to predict intercept probability from feature vectors.

| Function | Description |
|---|---|
| `train(data_path, model_path, n_samples, seed, generate_if_missing, verbose)` | Full training pipeline: loads or generates the MC dataset, splits into train/validation/test (70/15/15), trains XGBoost (300 estimators, max_depth=6, learning_rate=0.1), evaluates on the test set (AUC-ROC, accuracy, precision, recall, F1), displays feature importances, analyses calibration error, saves the model to JSON, and generates diagnostic plots. |
| `_plot_diagnostics(model, X_test, y_test, y_pred_proba, importances)` | Generates a 3-panel diagnostic figure: ROC curve with AUC, calibration curve, and feature importance bar chart. |

---

### `check_python.py` — Environment Diagnostic

Simple script that prints the Python version and executable path to the console. No functions — runs on import/execution.

---

### `app.py` — Streamlit Dashboard

Interactive web dashboard. The recommended way to run GeneralSim.

#### Sidebar configuration

| Section | Controls |
|---|---|
| **Radars** (1–12) | Position, frequency, TX power, antenna gain, beamwidth, look azimuth/elevation |
| **Targets** (1–20) | Position, velocity (Vx/Vy/Vz), RCS, **Target Type** (drone / helicopter / fixed_wing) |
| **Interceptor Systems** (1–12) | Name, position, max/min range, max velocity, reaction time, salvo size |
| **Processing Options** | Enable Clutter, Enable MTI Filter, **Enable Target Classification**, Use ML Model |
| **Time Stepping** | Simulation duration and time-step interval |

When **Target Type** is changed in a target expander, the velocity magnitude, RCS, and altitude automatically snap to realistic defaults for that type while preserving the direction of travel:

| Type | Typical speed | Typical RCS | Typical altitude |
|---|---|---|---|
| `drone` | 60 m/s | −5 dBsm | 200 m |
| `helicopter` | 80 m/s | 8 dBsm | 500 m |
| `fixed_wing` | 220 m/s | 12 dBsm | 2 000 m |

#### Tabs

| Tab | Content |
|---|---|
| **Scenario** | Animated Plotly tactical map (dark theme) with radar beam wedges, interceptor range rings, target trails, and play/pause/time-slider controls. Falls back to a static matplotlib preview before the simulation runs. |
| **Signal Processing** | Per-radar range-Doppler maps and CFAR detection tables for the selected timestep. |
| **Fusion** | Fused target table (position, velocity, track quality, **Type**, **Confidence**); **Classification Summary** panel (count and mean confidence per type); 17-feature vector inspection for each system–target pair. |
| **Intercept Assessment** | Analytical and ML intercept probability heatmaps; comparison scatter plot with correlation and mean/max difference metrics; mean P(intercept) over time chart. |
| **Recommendation** | Animated Plotly engagement map showing best-interceptor assignments with probability labels; summary table with Best Interceptor, P(intercept), and **Type** per fused target. |

#### Key helper functions

| Function | Description |
|---|---|
| `build_scenario_plotly(sim_results, systems)` | Builds an animated dark-theme Plotly tactical map with radar wedges, range rings, target trails, and velocity arrows. |
| `build_recommendation_plotly(sim_results, systems)` | Builds an animated Plotly map showing engagement lines and probability annotations between interceptors and fused targets. Hover text includes target type and classification confidence. |
| `plot_range_doppler(results, radar_idx)` | Returns a 2-panel matplotlib figure: Range-Doppler map and CFAR detections overlay. |
| `plot_intercept_heatmap(systems, fused_targets, P, title)` | Returns a RdYlGn heatmap of the (N_systems × N_targets) probability matrix. |
| `plot_comparison_scatter(P_ana, P_ml)` | Returns a scatter plot comparing analytical vs ML intercept probabilities. |

---

## Requirements

Install all dependencies with:

```bash
pip install numpy scipy matplotlib xgboost scikit-learn streamlit pandas plotly
```

| Package | Purpose |
|---|---|
| `numpy` | Array math throughout the pipeline |
| `scipy` | Least-squares solvers used in fusion |
| `matplotlib` | Static plot generation |
| `xgboost` | ML intercept probability model |
| `scikit-learn` | Train/test split and evaluation metrics |
| `streamlit` | Interactive dashboard (`app.py`) |
| `pandas` | DataFrames displayed in the dashboard |
| `plotly` | Animated interactive tactical maps in the dashboard |

---

## How to Run

### Option 1 — Streamlit Dashboard (recommended)

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

Configure radars, targets (including target type), and interceptor systems in the sidebar, then press **▶ Run Simulation**. Use the time slider to step through the scenario frames.

> If `streamlit` is not on your PATH, activate the virtual environment first:
> ```bash
> .venv\Scripts\activate
> streamlit run app.py
> ```

### Option 2 — Command-line script

```bash
python main.py
```

Runs the full pipeline headlessly and saves plots as PNG files in the working directory.

### Option 3 — Train the XGBoost model

```bash
python train_model.py
```

Generates Monte Carlo engagement data (if missing) and trains/saves `xgb_intercept_model.json`. This is required before the dashboard can use the ML model.
