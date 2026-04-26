from __future__ import annotations

from typing import Any

import numpy as np

OBSERVATION_SIZE = 13


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _normalized_time_to_collision(distance_px: float | None, speed_px_s: float, horizon_s: float = 2.5) -> float:
    if distance_px is None:
        return 1.0
    if speed_px_s <= 1e-6:
        return 1.0
    return _clip01((distance_px / max(1.0, speed_px_s)) / horizon_s)


def build_dino_observation(
    *,
    player_y_px: float,
    player_vy_px: float,
    grounded: bool,
    ducking: bool,
    current_speed_px_s: float,
    screen_width_px: float,
    screen_height_px: float,
    floor_y_px: float,
    max_speed_px_s: float,
    nearest_distance_px: float | None,
    nearest_width_px: float | None,
    nearest_height_px: float | None,
    nearest_is_bird: bool,
    second_distance_px: float | None = None,
    second_width_px: float | None = None,
    second_height_px: float | None = None,
    obstacle_count: int = 0,
) -> np.ndarray:
    player_y_norm = _clip01(player_y_px / max(1.0, floor_y_px))
    vy_norm = _clip01((player_vy_px + 2200.0) / 4400.0)
    grounded_value = 1.0 if grounded else 0.0
    ducking_value = 1.0 if ducking else 0.0
    speed_norm = _clip01(current_speed_px_s / max(1.0, max_speed_px_s))

    nearest_distance_norm = _clip01(nearest_distance_px / max(1.0, screen_width_px)) if nearest_distance_px is not None else 1.0
    nearest_ttc_norm = _normalized_time_to_collision(nearest_distance_px, current_speed_px_s)
    nearest_width_norm = _clip01(nearest_width_px / max(1.0, screen_width_px)) if nearest_width_px is not None else 0.0
    nearest_height_norm = _clip01(nearest_height_px / max(1.0, screen_height_px)) if nearest_height_px is not None else 0.0
    nearest_is_bird_value = 1.0 if nearest_is_bird else 0.0

    second_distance_norm = _clip01(second_distance_px / max(1.0, screen_width_px)) if second_distance_px is not None else 1.0
    second_ttc_norm = _normalized_time_to_collision(second_distance_px, current_speed_px_s)
    _ = second_width_px, second_height_px  # kept for future richer policies
    obstacle_count_norm = _clip01(obstacle_count / 4.0)

    return np.array(
        [
            player_y_norm,
            vy_norm,
            grounded_value,
            ducking_value,
            speed_norm,
            nearest_distance_norm,
            nearest_ttc_norm,
            nearest_width_norm,
            nearest_height_norm,
            nearest_is_bird_value,
            second_distance_norm,
            second_ttc_norm,
            obstacle_count_norm,
        ],
        dtype=np.float32,
    )
