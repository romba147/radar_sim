"""
fusion.py — Multi-Radar Detection Association & Fusion

Associates detections from multiple radars to physical targets,
then estimates Cartesian position (triangulation or boresight fallback)
and velocity vector.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from scenario import Radar, C


@dataclass
class FusedTarget:
    """A target estimate produced by fusing detections from one or more radars."""
    fused_index: int                          # sequential index in the fused list
    position: np.ndarray                      # estimated Cartesian [x, y, z] (m)
    velocity_vector: np.ndarray               # estimated Cartesian [vx, vy, vz] (m/s)
    n_radars: int                             # number of radars that detected this target
    radar_detections: List[Dict[str, Any]]    # per-radar detection info
    power_dB: float                           # best (highest) power across radars
    track_quality: float                      # quality metric [0, 1]
    position_method: str                      # "triangulation" or "boresight_approx"
    target_type: Optional[str] = None         # classifier output: "drone", "helicopter", "fixed_wing"
    classification_confidence: Optional[float] = None  # classifier confidence [0, 1]


def _boresight_direction(radar: Radar) -> np.ndarray:
    """Unit vector in boresight direction of the radar."""
    az = np.radians(radar.look_azimuth)
    el = np.radians(radar.look_elevation)
    return np.array([np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az),
                     np.sin(el)])


def _approx_cartesian(radar: Radar, detection: Dict[str, Any]) -> np.ndarray:
    """Approximate target Cartesian position: radar pos + range * boresight."""
    u = _boresight_direction(radar)
    return radar.position + detection["range"] * u


def _triangulate_position(radars: List[Radar],
                           detections: List[Dict[str, Any]],
                           radar_indices: List[int]) -> Optional[np.ndarray]:
    """Estimate Cartesian position by least-squares intersection of range spheres.

    Given N radars at positions p_i and measured ranges r_i, we solve:
        |x - p_i|^2 = r_i^2   for all i

    Subtract the last equation from each other to linearise:
        2 (p_N - p_i) . x = r_i^2 - r_N^2 - |p_i|^2 + |p_N|^2

    This gives a linear system  A x = b  solved via least-squares.
    """
    n = len(radar_indices)
    if n < 2:
        return None

    positions = [radars[ri].position for ri in radar_indices]
    ranges = [det["range"] for det in detections]

    # reference: last radar
    p_ref = positions[-1]
    r_ref = ranges[-1]

    A = np.zeros((n - 1, 3))
    b = np.zeros(n - 1)

    for k in range(n - 1):
        A[k] = 2.0 * (p_ref - positions[k])
        b[k] = (ranges[k]**2 - r_ref**2
                - np.dot(positions[k], positions[k])
                + np.dot(p_ref, p_ref))

    # check conditioning
    try:
        cond = np.linalg.cond(A)
        if cond > 1e6:
            return None  # ill-conditioned, fall back to boresight
        x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        return x
    except np.linalg.LinAlgError:
        return None


def _estimate_velocity_vector(radars: List[Radar],
                               detections: List[Dict[str, Any]],
                               radar_indices: List[int],
                               target_pos: np.ndarray) -> np.ndarray:
    """Estimate 3D velocity vector from radial velocities.

    Each radar measures v_r_i = v . u_i  where u_i is the unit LOS from
    radar i to the target.  Stack into  A v = v_r  and solve via
    least-squares.

    With only 1 radar, returns the velocity projected along the single LOS.
    With 2 radars in 3D, the system is under-determined — use least-norm solution.
    """
    n = len(radar_indices)

    A = np.zeros((n, 3))
    v_r = np.zeros(n)

    for k, (ri, det) in enumerate(zip(radar_indices, detections)):
        delta = target_pos - radars[ri].position
        dist = np.linalg.norm(delta)
        if dist > 0:
            A[k] = delta / dist
        v_r[k] = det["velocity"]

    try:
        vel, _, _, _ = np.linalg.lstsq(A, v_r, rcond=None)
        return vel
    except np.linalg.LinAlgError:
        return np.zeros(3)


def associate_and_fuse(radars: List[Radar],
                       per_radar_detections: Dict[int, List[Dict[str, Any]]],
                       gate_factor: float = 2.0) -> List[FusedTarget]:
    """Associate detections across radars and fuse into FusedTarget objects.

    Parameters
    ----------
    radars : list of Radar objects (indexed to match per_radar_detections keys)
    per_radar_detections : dict keyed by radar index, each value is a list
        of detection dicts (as returned by RadarProcessor.process()["estimated_targets"])
    gate_factor : multiplicative factor for the gating threshold
        (threshold = gate_factor * range * sin(beamwidth))

    Returns
    -------
    list of FusedTarget
    """
    # Step 1: compute approximate Cartesian for every detection
    cart_detections = []  # list of (radar_idx, det_idx, position_approx, detection_dict)
    for r_idx in sorted(per_radar_detections.keys()):
        radar = radars[r_idx]
        for d_idx, det in enumerate(per_radar_detections[r_idx]):
            pos = _approx_cartesian(radar, det)
            cart_detections.append((r_idx, d_idx, pos, det))

    if not cart_detections:
        return []

    # Step 2: greedy nearest-neighbor association
    n = len(cart_detections)
    assigned = [False] * n
    clusters = []  # each cluster: list of indices into cart_detections

    for i in range(n):
        if assigned[i]:
            continue

        r_idx_i, _, pos_i, det_i = cart_detections[i]
        radar_i = radars[r_idx_i]
        cluster = [i]
        assigned[i] = True

        gate_i = (gate_factor * det_i["range"]
                  * np.sin(np.radians(radar_i.antenna_beamwidth)))

        # track which radars are already in the cluster
        cluster_radars = {r_idx_i}

        for j in range(i + 1, n):
            if assigned[j]:
                continue

            r_idx_j, _, pos_j, det_j = cart_detections[j]

            # enforce at most one detection per radar per cluster
            if r_idx_j in cluster_radars:
                continue

            radar_j = radars[r_idx_j]
            gate_j = (gate_factor * det_j["range"]
                      * np.sin(np.radians(radar_j.antenna_beamwidth)))
            gate = max(gate_i, gate_j)

            dist = np.linalg.norm(pos_i - pos_j)
            if dist < gate:
                cluster.append(j)
                assigned[j] = True
                cluster_radars.add(r_idx_j)

        clusters.append(cluster)

    # Step 3: fuse each cluster
    fused_targets = []
    for fused_idx, cluster in enumerate(clusters):
        radar_indices = [cart_detections[k][0] for k in cluster]
        detections = [cart_detections[k][3] for k in cluster]
        unique_radars = list(set(radar_indices))
        n_radars = len(unique_radars)

        # per-radar detection info
        radar_det_info = []
        for k in cluster:
            r_idx, d_idx, _, det = cart_detections[k]
            radar_det_info.append({
                "radar_index": r_idx,
                "range": det["range"],
                "velocity": det["velocity"],
                "power_dB": det["power_dB"],
            })

        # position estimation
        if n_radars >= 2:
            # gather one detection per radar (best power if multiple from same radar)
            best_per_radar = {}
            for k in cluster:
                r_idx = cart_detections[k][0]
                det = cart_detections[k][3]
                if r_idx not in best_per_radar or det["power_dB"] > best_per_radar[r_idx][1]["power_dB"]:
                    best_per_radar[r_idx] = (k, det)

            tri_radar_indices = sorted(best_per_radar.keys())
            tri_detections = [best_per_radar[ri][1] for ri in tri_radar_indices]

            pos = _triangulate_position(radars, tri_detections, tri_radar_indices)
            if pos is not None:
                method = "triangulation"
            else:
                # fallback: average of approximate positions
                positions = [cart_detections[k][2] for k in cluster]
                pos = np.mean(positions, axis=0)
                method = "boresight_approx"
        else:
            pos = cart_detections[cluster[0]][2]
            method = "boresight_approx"

        # velocity vector estimation
        # use one detection per radar for the velocity solve
        if n_radars >= 2 and method == "triangulation":
            tri_radar_indices = sorted(set(radar_indices))
            tri_dets = []
            for ri in tri_radar_indices:
                # pick detection with best power from this radar
                best = max((cart_detections[k][3] for k in cluster
                            if cart_detections[k][0] == ri),
                           key=lambda d: d["power_dB"])
                tri_dets.append(best)
            vel = _estimate_velocity_vector(radars, tri_dets,
                                            tri_radar_indices, pos)
        else:
            # single radar: velocity along LOS only
            r_idx = radar_indices[0]
            delta = pos - radars[r_idx].position
            dist = np.linalg.norm(delta)
            if dist > 0:
                u_los = delta / dist
            else:
                u_los = np.array([1.0, 0.0, 0.0])
            vel = detections[0]["velocity"] * u_los

        # best power
        best_power = max(det["power_dB"] for det in detections)

        # track quality: higher for multi-radar, scaled by mean SNR
        mean_power = np.mean([det["power_dB"] for det in detections])
        snr_factor = np.clip((mean_power + 40) / 60, 0, 1)  # map [-40, 20] dB → [0, 1]
        quality = np.clip(snr_factor * np.sqrt(n_radars / 3.0), 0, 1)

        fused_targets.append(FusedTarget(
            fused_index=fused_idx,
            position=pos,
            velocity_vector=vel,
            n_radars=n_radars,
            radar_detections=radar_det_info,
            power_dB=best_power,
            track_quality=quality,
            position_method=method,
        ))

    return fused_targets


def print_fusion_report(fused_targets: List[FusedTarget]):
    """Print a summary table of fused targets."""
    print(f"\n  {len(fused_targets)} fused targets")
    print("-" * 110)
    print(f"  {'FT':>3s}  {'X [m]':>8s}  {'Y [m]':>8s}  {'Z [m]':>8s}  "
          f"{'|V| [m/s]':>10s}  {'Radars':>6s}  {'Quality':>7s}  {'Method':<16s}  "
          f"{'PdB':>6s}  {'Type':<12s}  {'Conf':>5s}")
    print("-" * 110)
    for ft in fused_targets:
        speed = np.linalg.norm(ft.velocity_vector)
        radar_str = ",".join(str(d["radar_index"]) for d in ft.radar_detections)
        ttype = ft.target_type or "unknown"
        conf = f"{ft.classification_confidence:.2f}" if ft.classification_confidence is not None else "  n/a"
        print(f"  {ft.fused_index:3d}  {ft.position[0]:8.0f}  {ft.position[1]:8.0f}  "
              f"{ft.position[2]:8.0f}  {speed:10.1f}  "
              f"{radar_str:>6s}  {ft.track_quality:7.3f}  "
              f"{ft.position_method:<16s}  {ft.power_dB:6.1f}  "
              f"{ttype:<12s}  {conf:>5s}")
    print("-" * 110)
