"""
feature_extraction.py — Unified feature engineering for the XGBoost intercept model.

Extracts the same feature vector for both:
  1. Training: from Monte Carlo dataset dicts
  2. Inference: from live InterceptorSystem + FusedTarget objects

Feature vector (17 features):
──────────────────────────────────────────────────────────────
 Geometric:
  0  distance            Euclidean dist from interceptor to target [m]
  1  closing_speed       radial velocity component toward interceptor [m/s]
  2  target_speed        magnitude of target velocity [m/s]
  3  azimuth_to_target   azimuth angle interceptor→target [deg]
  4  elevation_to_target elevation angle interceptor→target [deg]

 Radar / track:
  5  track_quality       fused track quality [0, 1]
  6  n_radars            number of radars tracking [1–3]
  7  power_dB            best signal power [dB]
  8  position_method     1 = triangulation, 0 = boresight_approx

 Interceptor system:
  9  max_range           [m]
 10  min_range           [m]
 11  max_target_velocity [m/s]
 12  reaction_time       [s]
 13  salvo_size          [#]

 Derived:
 14  range_fraction      (distance - min_range) / (max_range - min_range)
 15  speed_fraction      |closing_speed| / max_target_velocity
 16  in_envelope         1 if within range & speed limits, else 0
──────────────────────────────────────────────────────────────
"""

import numpy as np
from typing import List

FEATURE_NAMES = [
    "distance", "closing_speed", "target_speed",
    "azimuth_to_target", "elevation_to_target",
    "track_quality", "n_radars", "power_dB", "position_method",
    "max_range", "min_range", "max_target_velocity",
    "reaction_time", "salvo_size",
    "range_fraction", "speed_fraction", "in_envelope",
]

N_FEATURES = len(FEATURE_NAMES)


def extract_from_system_target(system, fused_target) -> np.ndarray:
    """Extract a 17-element feature vector from an InterceptorSystem and FusedTarget.

    Parameters
    ----------
    system : InterceptorSystem
    fused_target : FusedTarget

    Returns
    -------
    np.ndarray of shape (17,)
    """
    tgt_pos = np.asarray(fused_target.position, dtype=float)
    tgt_vel = np.asarray(fused_target.velocity_vector, dtype=float)
    sys_pos = np.asarray(system.position, dtype=float)

    delta = tgt_pos - sys_pos
    distance = np.linalg.norm(delta)
    target_speed = np.linalg.norm(tgt_vel)

    if distance > 0:
        u_to_int = -delta / distance
        closing_speed = float(np.dot(tgt_vel, u_to_int))
        u_los = delta / distance
        azimuth = np.degrees(np.arctan2(u_los[1], u_los[0]))
        elevation = np.degrees(np.arcsin(np.clip(u_los[2], -1, 1)))
    else:
        closing_speed = 0.0
        azimuth = 0.0
        elevation = 0.0

    # Derived
    range_span = system.max_range - system.min_range
    range_fraction = (distance - system.min_range) / range_span if range_span > 0 else 0.5
    speed_fraction = abs(closing_speed) / system.max_target_velocity if system.max_target_velocity > 0 else 0.0
    in_envelope = 1.0 if (system.min_range <= distance <= system.max_range and
                          abs(closing_speed) <= system.max_target_velocity) else 0.0

    position_method = 1.0 if fused_target.position_method == "triangulation" else 0.0

    return np.array([
        distance,
        closing_speed,
        target_speed,
        azimuth,
        elevation,
        fused_target.track_quality,
        float(fused_target.n_radars),
        fused_target.power_dB,
        position_method,
        system.max_range,
        system.min_range,
        system.max_target_velocity,
        system.reaction_time,
        float(system.salvo_size),
        range_fraction,
        speed_fraction,
        in_envelope,
    ], dtype=float)


def extract_matrix(systems, fused_targets) -> np.ndarray:
    """Extract feature matrix for all (system, target) pairs.

    Parameters
    ----------
    systems : list of InterceptorSystem
    fused_targets : list of FusedTarget

    Returns
    -------
    np.ndarray of shape (n_systems * n_targets, 17)
    Also returns index arrays for reconstructing the (i, j) mapping.
    """
    rows = []
    for sys in systems:
        for ft in fused_targets:
            rows.append(extract_from_system_target(sys, ft))
    return np.vstack(rows)


def dataset_to_Xy(data: dict):
    """Convert a Monte Carlo dataset dict to feature matrix X and label vector y.

    Parameters
    ----------
    data : dict from mc_simulator.generate_dataset() or load_dataset()

    Returns
    -------
    X : np.ndarray of shape (n_samples, 17)
    y : np.ndarray of shape (n_samples,)
    """
    n = len(data["hit"])
    X = np.column_stack([data[name] for name in FEATURE_NAMES])
    y = data["hit"].astype(float)
    return X, y
