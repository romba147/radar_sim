"""
classifier.py — Spectrogram-based airborne target classifier

Extracts micro-Doppler features from a Range-Doppler map at a detected target's
range bin and classifies the target as one of:
  - "drone"       (quadcopter / small UAV)
  - "helicopter"  (rotary-wing, slow main-rotor modulation)
  - "fixed_wing"  (aircraft / cruise missile, minimal micro-Doppler)
"""

import numpy as np
from typing import Dict, Tuple


def extract_doppler_features(rd_map: np.ndarray,
                              detection: Dict,
                              velocity_axis: np.ndarray) -> Dict:
    """Extract micro-Doppler spectral features at a detected target's range bin.

    Parameters
    ----------
    rd_map : complex Range-Doppler map (N_range x N_doppler)
    detection : dict with at least 'range_bin' and 'doppler_bin' keys
    velocity_axis : 1-D array of velocity values [m/s] for each Doppler bin

    Returns
    -------
    dict with feature values
    """
    range_bin = detection["range_bin"]
    doppler_bin = detection["doppler_bin"]

    # Doppler profile at the target's range (magnitude in dB)
    profile_mag = np.abs(rd_map[range_bin, :])
    profile_db = 20 * np.log10(profile_mag + 1e-30)
    profile_db -= profile_db.max()  # normalise to 0 dB peak

    peak_bin = doppler_bin
    peak_vel = velocity_axis[peak_bin]
    dv = velocity_axis[1] - velocity_axis[0]  # bin width [m/s]

    # --- 1. Doppler bandwidth (10-dB width) ---
    above_threshold = profile_db >= -10.0
    bandwidth_bins = int(above_threshold.sum())
    doppler_bandwidth_ms = bandwidth_bins * abs(dv)

    # --- 2. Sideband symmetry (ratio of energy on each side of the peak) ---
    left = profile_mag[:peak_bin]
    right = profile_mag[peak_bin + 1:]
    min_len = min(len(left), len(right))
    if min_len > 0:
        e_left = np.sum(left[-min_len:] ** 2)
        e_right = np.sum(right[:min_len] ** 2)
        total = e_left + e_right
        # 1.0 = perfectly symmetric, 0.0 = all energy on one side
        sideband_symmetry = 1.0 - abs(e_left - e_right) / (total + 1e-30)
    else:
        sideband_symmetry = 0.5

    # --- 3. Spectral entropy (spread of energy across Doppler bins) ---
    p = profile_mag ** 2
    p_norm = p / (p.sum() + 1e-30)
    spectral_entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-30)))
    # Normalise to [0, 1] relative to maximum possible entropy
    max_entropy = np.log(len(p_norm))
    spectral_entropy_norm = spectral_entropy / (max_entropy + 1e-30)

    # --- 4. Strongest secondary peak offset from bulk velocity ---
    # Suppress the main peak region (±3 bins) and find the next highest peak
    suppressed = profile_db.copy()
    half_guard = 3
    lo = max(0, peak_bin - half_guard)
    hi = min(len(suppressed), peak_bin + half_guard + 1)
    suppressed[lo:hi] = -100.0

    secondary_bin = int(np.argmax(suppressed))
    peak_sideband_offset_ms = abs(velocity_axis[secondary_bin] - peak_vel)

    return {
        "doppler_bandwidth_ms": doppler_bandwidth_ms,
        "sideband_symmetry": sideband_symmetry,
        "spectral_entropy": spectral_entropy_norm,
        "peak_sideband_offset_ms": peak_sideband_offset_ms,
    }


def classify_target(features: Dict) -> Tuple[str, float]:
    """Rule-based target classifier using micro-Doppler spectral features.

    Decision logic mirrors the physics described in the airborne-target-
    classification literature:

    Drone       — wide, symmetric sidebands from fast-spinning rotors
                  (>200 Hz flash rate → wide Doppler spread, high entropy)
    Helicopter  — moderate spread, dominant low-frequency sideband from the
                  slow main rotor (~10 Hz)
    Fixed-wing  — narrow bulk-Doppler return, low entropy, minimal sidebands

    Parameters
    ----------
    features : dict returned by extract_doppler_features()

    Returns
    -------
    (label, confidence) — label in {"drone", "helicopter", "fixed_wing"},
                          confidence in [0, 1]
    """
    bw = features["doppler_bandwidth_ms"]
    sym = features["sideband_symmetry"]
    ent = features["spectral_entropy"]
    sideband_off = features["peak_sideband_offset_ms"]

    # ── Score each class ────────────────────────────────────────────────
    scores = {}

    # Drone: high bandwidth + high symmetry + high entropy
    drone_bw_score = np.clip((bw - 5.0) / 20.0, 0.0, 1.0)
    drone_sym_score = np.clip((sym - 0.5) / 0.5, 0.0, 1.0)
    drone_ent_score = np.clip((ent - 0.4) / 0.4, 0.0, 1.0)
    scores["drone"] = (drone_bw_score * 0.4
                       + drone_sym_score * 0.3
                       + drone_ent_score * 0.3)

    # Helicopter: moderate bandwidth, small sideband offset (slow rotor)
    heli_bw_score = np.clip(1.0 - abs(bw - 8.0) / 8.0, 0.0, 1.0)
    heli_off_score = np.clip(1.0 - (sideband_off - 1.0) / 15.0, 0.0, 1.0)
    heli_ent_score = np.clip((ent - 0.2) / 0.4, 0.0, 1.0)
    scores["helicopter"] = (heli_bw_score * 0.35
                            + heli_off_score * 0.35
                            + heli_ent_score * 0.30)

    # Fixed-wing: narrow bandwidth, low entropy
    fw_bw_score = np.clip(1.0 - bw / 10.0, 0.0, 1.0)
    fw_ent_score = np.clip(1.0 - ent / 0.4, 0.0, 1.0)
    scores["fixed_wing"] = fw_bw_score * 0.5 + fw_ent_score * 0.5

    # ── Pick winner and compute confidence ──────────────────────────────
    label = max(scores, key=scores.__getitem__)
    raw_score = scores[label]

    total = sum(scores.values())
    confidence = float(raw_score / (total + 1e-30))
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return label, confidence
