from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DinoVisionConfig:
    playfield_top_ratio: float = 0.12
    playfield_bottom_ratio: float = 1.0
    playfield_left_ratio: float = 0.0
    playfield_right_ratio: float = 1.0
    dark_threshold: int = 195
    light_threshold: int = 70
    min_obstacle_area: int = 120
    min_player_area: int = 80
    min_player_width_px: int = 24
    min_player_height_px: int = 24
    max_player_width_px: int = 95
    max_player_height_px: int = 95
    max_player_aspect_ratio: float = 2.8
    max_standing_player_width_px: int = 70
    min_standing_player_height_px: int = 56
    max_ducking_player_height_px: int = 45
    max_player_x_px: int = 190
    max_player_x_fraction: float = 0.22
    max_player_airborne_gap_px: int = 130
    max_player_center_jump_px: int = 45
    max_player_vertical_jump_px: int = 90
    player_ground_band_px: int = 80
    player_search_fraction: float = 0.33
    ground_search_fraction: float = 0.58
    obstacle_top_margin_px: int = 120
    obstacle_side_margin_px: int = 4
    obstacle_player_clearance_px: int = 12
    ground_line_cleanup_px: int = 4
    max_obstacle_width_fraction: float = 0.22
    bird_height_threshold_px: int = 14
    min_bird_width_px: int = 24
    min_bird_height_px: int = 18
    speed_smoothing: float = 0.35
    jump_base_distance_px: float = 72.0
    jump_speed_factor: float = 0.12
    duck_distance_px: float = 50.0
    min_action_interval_s: float = 0.18


@dataclass(slots=True)
class RealGameConfig:
    window_title_candidates: tuple[str, ...] = ("dino", "Chrome")
    manual_region: tuple[int, int, int, int] | None = None
    frame_rate: float = 50.0
    debug: bool = True
    auto_start: bool = True
    auto_restart: bool = True
    focus_window: bool = True
    keep_window_foreground: bool = True
    focus_refresh_s: float = 2.0


@dataclass(slots=True)
class SimConfig:
    screen_width: int = 600
    screen_height: int = 200
    ground_height: int = 36
    step_interval_s: float = 1.0 / 30.0
    physics_step_interval_s: float = 1.0 / 60.0
    player_x: int = 70
    player_width: int = 30
    stand_height: int = 47
    duck_height: int = 25
    gravity: float = 2160.0
    jump_velocity: float = 600.0
    base_speed: float = 360.0
    max_speed: float = 780.0
    speed_acceleration: float = 3.6
    speed_increment: float = 7.5
    survival_reward: float = 0.03
    pass_reward: float = 0.7
    collision_penalty: float = -1.0
    max_episode_steps: int = 3000
    initial_obstacle_delay_s: float = 3.0
    min_spawn_gap_px: int = 240
    max_spawn_gap_px: int = 520
    gap_coefficient: float = 0.6
    bird_probability: float = 0.18
    domain_randomization: bool = True
    domain_randomization_strength: float = 0.12
    observation_noise: float = 0.02
    action_latency_steps: int = 0
    observation_latency_steps: int = 0
    survival_reward: float = 0.03
    pass_reward: float = 0.7
    collision_penalty: float = -1.0
    action_penalty: float = -0.002
    unnecessary_jump_penalty: float = -0.05
    unnecessary_duck_penalty: float = -0.06
    missed_action_penalty: float = -0.15
    teacher_match_reward: float = 0.08
    teacher_mismatch_penalty: float = -0.08
    enable_teacher_shaping: bool = True
    seed: int = 42


@dataclass(slots=True)
class NeatConfig:
    generations: int = 20
    population: int = 24
    episode_distance_px: float = 2400.0
    survival_reward: float = 0.005
    pass_reward: float = 3.0
    collision_penalty: float = -2.0
    action_penalty: float = -0.002
    unnecessary_jump_penalty: float = -0.05
    unnecessary_duck_penalty: float = -0.04
    missed_action_penalty: float = -0.5
    teacher_match_reward: float = 0.08
    teacher_mismatch_penalty: float = -0.25
    safety_override_penalty: float = -0.08
    enable_safety_overrides: bool = True
    pass_detection_distance_px: float = 90.0
    pass_confirmation_distance_px: float = 140.0
    pass_cross_margin_px: float = 18.0
    safe_jump_distance_px: float = 140.0
    duck_bird_distance_px: float = 120.0
    max_episode_frames: int = 3000
    stalled_death_seconds: float = 0.45
    stalled_speed_px_s: float = 25.0
    pass_confirmation_seconds: float = 0.35
    reset_timeout_seconds: float = 3.0
    reset_poll_seconds: float = 0.08
    observation_size: int = 13
    seed: int = 42
    teacher_source: str | None = None
    output_dir: Path = field(default_factory=lambda: Path("artifacts") / "neat")




