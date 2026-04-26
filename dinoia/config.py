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
    player_x: int = 70
    player_width: int = 30
    stand_height: int = 44
    duck_height: int = 28
    gravity: float = 2200.0
    jump_velocity: float = 780.0
    base_speed: float = 320.0
    max_speed: float = 920.0
    speed_increment: float = 7.5
    survival_reward: float = 0.03
    pass_reward: float = 0.7
    collision_penalty: float = -1.0
    max_episode_steps: int = 3000
    min_spawn_gap_px: int = 240
    max_spawn_gap_px: int = 520
    bird_probability: float = 0.18
    domain_randomization: bool = True
    domain_randomization_strength: float = 0.12
    observation_noise: float = 0.02
    seed: int = 42


@dataclass(slots=True)
class TrainingConfig:
    total_timesteps: int = 120_000
    preset: str = "standard"
    resume: str = "auto"
    fresh: bool = False
    device: str = "auto"
    learning_rate: float = 3e-4
    gamma: float = 0.995
    buffer_size: int = 200_000
    learning_starts: int = 5_000
    batch_size: int = 128
    train_freq: int = 4
    target_update_interval: int = 2_500
    exploration_fraction: float = 0.35
    exploration_final_eps: float = 0.02
    n_eval_episodes: int = 10
    n_envs: int = 4
    observation_size: int = 13
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("artifacts") / "dqn")




