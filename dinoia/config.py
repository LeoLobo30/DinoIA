from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SimConfig:
    screen_width: int = 600
    screen_height: int = 150
    ground_height: int = 10
    step_interval_s: float = 1.0 / 30.0
    physics_step_interval_s: float = 1.0 / 60.0
    target_fps: float = 60.0
    frame_time_jitter: float = 0.1
    frame_time_spike_probability: float = 0.03
    frame_time_spike_scale: float = 1.8
    player_x: int = 50
    player_width: int = 44
    duck_width: int = 59
    stand_height: int = 47
    duck_height: int = 25
    gravity: float = 2160.0
    jump_velocity: float = 600.0
    base_speed: float = 360.0
    max_speed: float = 780.0
    speed_acceleration: float = 60.0
    speed_increment: float = 7.5
    speed_unit_px_s: float = 60.0
    speed_increment_interval_score: float = 100.0
    distance_scale_per_step: float = 0.025
    curriculum_learning: bool = False
    curriculum_low_episodes: int = 1_000
    curriculum_medium_episodes: int = 3_000
    curriculum_high_episodes: int = 6_000
    curriculum_low_speed_min: float = 360.0
    curriculum_low_speed_max: float = 480.0
    curriculum_medium_speed_min: float = 480.0
    curriculum_medium_speed_max: float = 660.0
    curriculum_high_speed_min: float = 660.0
    curriculum_high_speed_max: float = 900.0
    curriculum_random_speed_min: float = 360.0
    curriculum_random_speed_max: float = 1080.0
    max_episode_steps: int = 3000
    initial_obstacle_delay_s: float = 3.0
    min_spawn_gap_px: int = 120
    max_spawn_gap_px: int = 420
    gap_coefficient: float = 0.6
    bird_probability: float = 0.2
    bird_min_speed_units: float = 8.5
    max_same_obstacle_streak: int = 2
    max_bird_streak: int = 1
    prohibit_bird_after_bird: bool = True
    max_obstacles_on_screen: int = 3
    recent_obstacle_memory: int = 4
    cactus_cluster_min_speed_units: float = 4.5
    cactus_cluster_high_speed_units: float = 7.0
    cactus_cluster_max_len_mid_speed: int = 2
    cactus_cluster_max_len_high_speed: int = 3
    domain_randomization: bool = False
    domain_randomization_strength: float = 0.12
    observation_noise: float = 0.02
    action_latency_steps: int = 0
    observation_latency_steps: int = 0
    survival_reward: float = 0.03
    pass_reward: float = 1.0
    speed_pass_reward_scale: float = 0.0
    collision_penalty: float = -1.0
    action_penalty: float = -0.002
    unnecessary_jump_penalty: float = -0.05
    unnecessary_duck_penalty: float = -0.06
    seed: int = 42

