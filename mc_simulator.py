"""
mc_simulator.py — Monte Carlo Engagement Simulator

Generates labeled training data for the XGBoost intercept probability model.
Each sample is a randomized (interceptor_system, target) engagement with
stochastic evasive maneuvers.  The outcome is binary: hit (1) or miss (0).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EngagementConfig:
    """Tunables for the Monte Carlo engagement engine."""
    # Interceptor kinematic model
    interceptor_speed_factor: float = 2.5   # interceptor speed = factor × max_target_velocity
    lethal_radius_base: float = 15.0        # base lethal radius [m]
    lethal_radius_range_scale: float = 0.002  # additional radius per m of max_range

    # Stochastic target maneuver model
    max_evasive_g: float = 5.0              # maximum evasive g-load
    min_evasive_g: float = 1.0              # minimum evasive g-load
    maneuver_probability: float = 0.7       # probability that target maneuvers at all
    max_maneuver_count: int = 3             # max number of maneuvers during intercept

    # Navigation model noise
    guidance_noise_std: float = 0.05        # proportional nav noise (fraction of closing distance)

    # Scenario randomization bounds
    range_min: float = 200.0                # min target distance from interceptor [m]
    range_max: float = 15000.0              # max target distance from interceptor [m]
    target_speed_min: float = 5.0           # min target speed [m/s]
    target_speed_max: float = 60.0          # max target speed [m/s]
    target_alt_max: float = 500.0           # max target altitude [m]

    # Track quality randomization
    track_quality_min: float = 0.1
    track_quality_max: float = 1.0
    n_radars_choices: List[int] = None      # defaults to [1, 2, 3]
    power_dB_min: float = -30.0
    power_dB_max: float = 20.0

    # Simulation time step
    dt: float = 1.0                         # [s]

    def __post_init__(self):
        if self.n_radars_choices is None:
            self.n_radars_choices = [1, 2, 3]


def _propagate_target(pos: np.ndarray, vel: np.ndarray,
                      duration: float, cfg: EngagementConfig,
                      rng: np.random.Generator) -> np.ndarray:
    """Propagate a target with stochastic evasive maneuvers.

    Returns the trajectory as an (N_steps, 3) array of positions.
    """
    dt = cfg.dt
    n_steps = max(int(duration / dt), 1)
    n_steps = min(n_steps, 10000)  # cap to prevent memory issues
    trajectory = np.zeros((n_steps + 1, 3))
    trajectory[0] = pos.copy()

    current_vel = vel.copy()
    speed = np.linalg.norm(current_vel)

    # Decide if target maneuvers at all
    will_maneuver = rng.random() < cfg.maneuver_probability and speed > 1e-3

    # Schedule random maneuver events
    maneuver_times = []
    if will_maneuver:
        n_maneuvers = rng.integers(1, cfg.max_maneuver_count + 1)
        maneuver_times = sorted(rng.uniform(0, duration, size=n_maneuvers))

    maneuver_idx = 0

    for step in range(n_steps):
        t = step * dt

        # Check if a maneuver triggers at this time step
        if maneuver_idx < len(maneuver_times) and t >= maneuver_times[maneuver_idx]:
            maneuver_idx += 1
            # Apply random turn: change heading by a random angle
            g_load = rng.uniform(cfg.min_evasive_g, cfg.max_evasive_g)
            turn_accel = g_load * 9.81  # m/s²

            # Random turn direction in the horizontal plane
            if speed > 1e-3:
                # Perpendicular direction to current velocity in x-y plane
                perp = np.array([-current_vel[1], current_vel[0], 0.0])
                perp_norm = np.linalg.norm(perp)
                if perp_norm > 1e-6:
                    perp /= perp_norm
                else:
                    perp = np.array([1.0, 0.0, 0.0])

                # Random sign
                sign = rng.choice([-1.0, 1.0])
                dv = sign * perp * turn_accel * dt * rng.uniform(2.0, 8.0)
                current_vel += dv

                # Maintain original speed (maneuver changes direction, not speed)
                new_speed = np.linalg.norm(current_vel)
                if new_speed > 1e-3:
                    current_vel = current_vel * (speed / new_speed)

        # Propagate position
        trajectory[step + 1] = trajectory[step] + current_vel * dt

    return trajectory


def _simulate_intercept(interceptor_pos: np.ndarray,
                        interceptor_speed: float,
                        lethal_radius: float,
                        max_flight_range: float,
                        target_trajectory: np.ndarray,
                        dt: float,
                        guidance_noise_std: float,
                        rng: np.random.Generator) -> float:
    """Simulate interceptor flying toward a maneuvering target.

    Uses proportional navigation with noise.
    The interceptor runs out of fuel after traveling max_flight_range.
    Returns the minimum distance achieved (miss distance).
    """
    int_pos = interceptor_pos.copy()
    start_pos = interceptor_pos.copy()
    n_steps = len(target_trajectory) - 1
    min_dist = np.inf

    for step in range(n_steps):
        tgt_pos = target_trajectory[step]
        delta = tgt_pos - int_pos
        dist = np.linalg.norm(delta)

        if dist < min_dist:
            min_dist = dist

        # Interceptor reached target
        if dist < lethal_radius:
            return dist

        if dist < 1e-3:
            return 0.0

        # Check fuel: has interceptor exceeded its max flight range?
        flown = np.linalg.norm(int_pos - start_pos)
        if flown > max_flight_range:
            return min_dist

        # Proportional navigation: fly toward target with noise
        u_los = delta / dist
        noise = rng.normal(0, guidance_noise_std, size=3)
        u_guided = u_los + noise
        u_norm = np.linalg.norm(u_guided)
        if u_norm > 1e-6:
            u_guided /= u_norm
        else:
            u_guided = u_los

        # Move interceptor
        step_dist = interceptor_speed * dt
        new_pos = int_pos + u_guided * step_dist

        # Compute closest point of approach along this step segment
        seg = new_pos - int_pos
        seg_len = np.linalg.norm(seg)
        if seg_len > 1e-6:
            t_cpa = np.dot(tgt_pos - int_pos, seg) / (seg_len * seg_len)
            t_cpa = np.clip(t_cpa, 0.0, 1.0)
            closest_pt = int_pos + t_cpa * seg
            cpa_dist = np.linalg.norm(tgt_pos - closest_pt)
            min_dist = min(min_dist, cpa_dist)

            if cpa_dist <= lethal_radius:
                return cpa_dist

        int_pos = new_pos

        # Check overshoot: if interceptor passed target
        if step_dist > dist:
            return min_dist

    # Final distance check
    dist = np.linalg.norm(target_trajectory[-1] - int_pos)
    min_dist = min(min_dist, dist)

    return min_dist


def _single_engagement(system_params: dict, target_params: dict,
                       cfg: EngagementConfig,
                       rng: np.random.Generator) -> dict:
    """Run one Monte Carlo engagement and return feature dict + outcome.

    Parameters
    ----------
    system_params : dict with keys matching InterceptorSystem fields
    target_params : dict with position, velocity_vector, track_quality, etc.
    cfg : EngagementConfig
    rng : numpy random generator

    Returns
    -------
    dict with all features and 'hit' label (0 or 1)
    """
    # Extract system and target info
    sys_pos = np.asarray(system_params["position"], dtype=float)
    tgt_pos = np.asarray(target_params["position"], dtype=float)
    tgt_vel = np.asarray(target_params["velocity_vector"], dtype=float)

    max_range = system_params["max_range"]
    min_range = system_params["min_range"]
    max_target_vel = system_params["max_target_velocity"]
    reaction_time = system_params["reaction_time"]
    salvo_size = system_params["salvo_size"]

    # Geometric features
    delta = tgt_pos - sys_pos
    distance = np.linalg.norm(delta)
    target_speed = np.linalg.norm(tgt_vel)

    if distance > 0:
        u_to_interceptor = -delta / distance
        closing_speed = float(np.dot(tgt_vel, u_to_interceptor))
    else:
        u_to_interceptor = np.array([1.0, 0.0, 0.0])
        closing_speed = 0.0

    # Azimuth and elevation from interceptor to target
    if distance > 0:
        u_los = delta / distance
        azimuth = np.degrees(np.arctan2(u_los[1], u_los[0]))
        elevation = np.degrees(np.arcsin(np.clip(u_los[2], -1, 1)))
    else:
        azimuth = 0.0
        elevation = 0.0

    # Derived features
    range_span = max_range - min_range
    range_fraction = (distance - min_range) / range_span if range_span > 0 else 0.5
    speed_fraction = abs(closing_speed) / max_target_vel if max_target_vel > 0 else 0.0
    in_envelope = 1.0 if (min_range <= distance <= max_range and
                          abs(closing_speed) <= max_target_vel) else 0.0

    # Track quality and radar features
    track_quality = target_params.get("track_quality", 0.5)
    n_radars = target_params.get("n_radars", 1)
    power_dB = target_params.get("power_dB", 0.0)
    position_method = 1.0 if target_params.get("position_method", "boresight_approx") == "triangulation" else 0.0

    # Interceptor speed model
    interceptor_speed = cfg.interceptor_speed_factor * max_target_vel

    # Lethal radius (larger systems have larger warheads)
    lethal_radius = cfg.lethal_radius_base + cfg.lethal_radius_range_scale * max_range

    # Max flight range: interceptor can fly up to 1.2× system max range before fuel depletion
    max_flight_range = 1.2 * max_range

    # Degrade guidance based on track quality (lower quality = more noise)
    quality_noise = cfg.guidance_noise_std * (1.0 + 2.0 * (1.0 - track_quality))

    # Estimate flight duration: reaction + time to reach target
    if interceptor_speed > 0:
        flight_time = distance / interceptor_speed
    else:
        flight_time = 999.0
    total_time = min(reaction_time + flight_time, 300.0)  # cap at 5 minutes

    # Propagate target with stochastic maneuvers for the full engagement time
    target_traj = _propagate_target(tgt_pos, tgt_vel, total_time, cfg, rng)

    # Target position at time of interceptor launch (after reaction time)
    reaction_steps = max(int(reaction_time / cfg.dt), 0)
    if reaction_steps < len(target_traj):
        tgt_at_launch = target_traj[reaction_steps]
    else:
        tgt_at_launch = target_traj[-1]

    # Simulate the salvo: each round is independent
    hit = 0
    for _ in range(salvo_size):
        # Re-propagate target (different random maneuvers per round)
        tgt_traj_round = _propagate_target(tgt_pos, tgt_vel, total_time, cfg, rng)
        launch_traj = tgt_traj_round[reaction_steps:] if reaction_steps < len(tgt_traj_round) else tgt_traj_round[-1:]
        miss_dist = _simulate_intercept(
            sys_pos, interceptor_speed, lethal_radius, max_flight_range,
            launch_traj, cfg.dt, quality_noise, rng
        )
        if miss_dist <= lethal_radius:
            hit = 1
            break

    # Build output
    return {
        # Features
        "distance": distance,
        "closing_speed": closing_speed,
        "target_speed": target_speed,
        "azimuth_to_target": azimuth,
        "elevation_to_target": elevation,
        "track_quality": track_quality,
        "n_radars": n_radars,
        "power_dB": power_dB,
        "position_method": position_method,
        "max_range": max_range,
        "min_range": min_range,
        "max_target_velocity": max_target_vel,
        "reaction_time": reaction_time,
        "salvo_size": salvo_size,
        "range_fraction": range_fraction,
        "speed_fraction": speed_fraction,
        "in_envelope": in_envelope,
        # Label
        "hit": hit,
    }


def _random_system(rng: np.random.Generator, reference_systems: list) -> dict:
    """Sample a random interceptor system configuration.

    Uses one of the reference systems as a base and adds small perturbations.
    """
    base = rng.choice(reference_systems)
    # Small perturbations (±10%) to create diversity
    jitter = lambda v, frac=0.1: v * rng.uniform(1 - frac, 1 + frac)

    pos = np.array(base["position"], dtype=float)
    pos[:2] += rng.uniform(-500, 500, size=2)  # spatial jitter

    return {
        "position": pos,
        "max_range": jitter(base["max_range"]),
        "min_range": jitter(base["min_range"]),
        "max_target_velocity": jitter(base["max_target_velocity"]),
        "reaction_time": jitter(base["reaction_time"]),
        "salvo_size": base["salvo_size"],  # keep integer
    }


def _random_target(rng: np.random.Generator, sys_pos: np.ndarray,
                   cfg: EngagementConfig) -> dict:
    """Generate a random target relative to a system position."""
    # Random direction in upper hemisphere
    azimuth = rng.uniform(-180, 180)
    elevation = rng.uniform(-10, 30)

    # Random range
    distance = rng.uniform(cfg.range_min, cfg.range_max)

    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    offset = distance * np.array([
        np.cos(el_rad) * np.cos(az_rad),
        np.cos(el_rad) * np.sin(az_rad),
        np.sin(el_rad),
    ])

    pos = sys_pos + offset

    # Random velocity
    speed = rng.uniform(cfg.target_speed_min, cfg.target_speed_max)
    vel_az = rng.uniform(-180, 180)
    vel_el = rng.uniform(-5, 5)
    vel_az_rad = np.radians(vel_az)
    vel_el_rad = np.radians(vel_el)
    vel = speed * np.array([
        np.cos(vel_el_rad) * np.cos(vel_az_rad),
        np.cos(vel_el_rad) * np.sin(vel_az_rad),
        np.sin(vel_el_rad),
    ])

    # Random track quality and radar features
    track_quality = rng.uniform(cfg.track_quality_min, cfg.track_quality_max)
    n_radars = int(rng.choice(cfg.n_radars_choices))
    power_dB = rng.uniform(cfg.power_dB_min, cfg.power_dB_max)

    # Higher n_radars → more likely to have triangulation
    position_method = "triangulation" if n_radars >= 2 and rng.random() > 0.2 else "boresight_approx"

    return {
        "position": pos,
        "velocity_vector": vel,
        "track_quality": track_quality,
        "n_radars": n_radars,
        "power_dB": power_dB,
        "position_method": position_method,
    }


# ── Reference interceptor system configs (same as main.py) ──────────

DEFAULT_REFERENCE_SYSTEMS = [
    {
        "position": [1000, 500, 0],
        "min_range": 500, "max_range": 4000,
        "max_target_velocity": 20.0, "reaction_time": 5.0, "salvo_size": 2,
    },
    {
        "position": [0, 1500, 0],
        "min_range": 1000, "max_range": 7000,
        "max_target_velocity": 35.0, "reaction_time": 8.0, "salvo_size": 1,
    },
    {
        "position": [-1000, 2000, 0],
        "min_range": 3000, "max_range": 10000,
        "max_target_velocity": 50.0, "reaction_time": 15.0, "salvo_size": 1,
    },
    {
        "position": [500, 0, 0],
        "min_range": 100, "max_range": 2000,
        "max_target_velocity": 40.0, "reaction_time": 2.0, "salvo_size": 4,
    },
]


def generate_dataset(n_samples: int = 50000,
                     reference_systems: list = None,
                     cfg: EngagementConfig = None,
                     seed: int = 42,
                     verbose: bool = True) -> dict:
    """Generate the full Monte Carlo training dataset.

    Parameters
    ----------
    n_samples : int
        Number of engagement samples to generate.
    reference_systems : list of dict
        Base interceptor system configurations to sample from.
    cfg : EngagementConfig
        Simulation parameters.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys = feature/label names, values = np.ndarray of length n_samples.
    """
    if reference_systems is None:
        reference_systems = DEFAULT_REFERENCE_SYSTEMS
    if cfg is None:
        cfg = EngagementConfig()

    rng = np.random.default_rng(seed)

    # Column names
    columns = [
        "distance", "closing_speed", "target_speed",
        "azimuth_to_target", "elevation_to_target",
        "track_quality", "n_radars", "power_dB", "position_method",
        "max_range", "min_range", "max_target_velocity",
        "reaction_time", "salvo_size",
        "range_fraction", "speed_fraction", "in_envelope",
        "hit",
    ]

    data = {col: np.zeros(n_samples) for col in columns}

    report_interval = max(n_samples // 10, 1)

    for i in range(n_samples):
        # Random system and target
        sys_params = _random_system(rng, reference_systems)
        tgt_params = _random_target(rng, np.asarray(sys_params["position"]), cfg)

        # Run engagement
        result = _single_engagement(sys_params, tgt_params, cfg, rng)

        for col in columns:
            data[col][i] = result[col]

        if verbose and (i + 1) % report_interval == 0:
            hit_rate = data["hit"][:i+1].mean() * 100
            print(f"  [{i+1:>6d}/{n_samples}]  hit rate so far: {hit_rate:.1f}%",
                  flush=True)

    return data


def save_dataset(data: dict, path: str = "mc_engagement_data.npz"):
    """Save the dataset to a compressed NPZ file."""
    np.savez_compressed(path, **data)
    total = len(data["hit"])
    hits = int(data["hit"].sum())
    print(f"  Saved {total} samples ({hits} hits, {total-hits} misses) → {path}")


def load_dataset(path: str = "mc_engagement_data.npz") -> dict:
    """Load a dataset from NPZ file."""
    loaded = np.load(path)
    return {key: loaded[key] for key in loaded.files}


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("    MONTE CARLO ENGAGEMENT SIMULATOR")
    print("=" * 62)

    cfg = EngagementConfig()
    data = generate_dataset(n_samples=50000, cfg=cfg, seed=42, verbose=True)
    save_dataset(data, "mc_engagement_data.npz")

    # Quick stats
    hits = data["hit"].sum()
    total = len(data["hit"])
    print(f"\n  Overall hit rate: {hits/total*100:.1f}%")
    print(f"  In-envelope samples: {data['in_envelope'].sum():.0f} / {total}")
    in_env = data["in_envelope"] == 1.0
    if in_env.sum() > 0:
        print(f"  In-envelope hit rate: {data['hit'][in_env].mean()*100:.1f}%")
    out_env = data["in_envelope"] == 0.0
    if out_env.sum() > 0:
        print(f"  Out-of-envelope hit rate: {data['hit'][out_env].mean()*100:.1f}%")
