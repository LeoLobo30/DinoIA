from __future__ import annotations

import cv2
import numpy as np

from dinoia.config import DinoVisionConfig
from dinoia.vision import DinoVisionAnalyzer


def _build_frame(offset: int = 0) -> np.ndarray:
    frame = np.full((220, 420, 3), 255, dtype=np.uint8)
    cv2.line(frame, (0, 176), (419, 176), (0, 0, 0), 2)
    cv2.rectangle(frame, (42, 128), (74, 168), (0, 0, 0), -1)
    cv2.rectangle(frame, (220 - offset, 122), (250 - offset, 168), (0, 0, 0), -1)
    return frame


def test_detector_finds_obstacle_and_estimates_speed():
    analyzer = DinoVisionAnalyzer(
        DinoVisionConfig(
            playfield_top_ratio=0.0,
            playfield_bottom_ratio=1.0,
            player_search_fraction=0.45,
            min_player_area=40,
            min_obstacle_area=80,
        )
    )

    state1 = analyzer.analyze(_build_frame(offset=0), timestamp=1.0)
    state2 = analyzer.analyze(_build_frame(offset=12), timestamp=1.1)

    assert state1.player_box.area > 0
    assert state1.nearest_obstacle is not None
    assert state1.nearest_obstacle.label == "cactus"
    assert state1.nearest_distance_px is not None
    assert state2.estimated_speed_px_s >= 0.0
    assert np.isfinite(state2.estimated_player_vy_px_s)


def test_detector_does_not_treat_player_as_nearest_obstacle():
    analyzer = DinoVisionAnalyzer(
        DinoVisionConfig(
            playfield_top_ratio=0.0,
            playfield_bottom_ratio=1.0,
            player_search_fraction=0.45,
            min_player_area=40,
            min_obstacle_area=40,
        )
    )

    frame = np.full((220, 420, 3), 255, dtype=np.uint8)
    cv2.line(frame, (0, 176), (419, 176), (0, 0, 0), 2)
    cv2.rectangle(frame, (42, 128), (74, 168), (0, 0, 0), -1)

    state = analyzer.analyze(frame, timestamp=1.0)

    assert state.player_box.area > 0
    assert state.nearest_obstacle is None


def test_detector_ignores_ground_line_connected_to_obstacle():
    analyzer = DinoVisionAnalyzer(
        DinoVisionConfig(
            playfield_top_ratio=0.0,
            playfield_bottom_ratio=1.0,
            player_search_fraction=0.45,
            min_player_area=40,
            min_obstacle_area=40,
        )
    )

    frame = np.full((220, 420, 3), 255, dtype=np.uint8)
    cv2.line(frame, (0, 176), (419, 176), (0, 0, 0), 2)
    cv2.rectangle(frame, (42, 128), (74, 168), (0, 0, 0), -1)
    cv2.rectangle(frame, (220, 128), (250, 168), (0, 0, 0), -1)

    state = analyzer.analyze(frame, timestamp=1.0)

    assert state.nearest_obstacle is not None
    assert state.nearest_obstacle.rect.w < 80
    assert state.nearest_distance_px > 100


def test_detector_ignores_score_like_tiny_birds():
    analyzer = DinoVisionAnalyzer(
        DinoVisionConfig(
            playfield_top_ratio=0.0,
            playfield_bottom_ratio=1.0,
            player_search_fraction=0.45,
            min_player_area=40,
            min_obstacle_area=40,
        )
    )

    frame = np.full((220, 420, 3), 255, dtype=np.uint8)
    cv2.line(frame, (0, 176), (419, 176), (0, 0, 0), 2)
    cv2.rectangle(frame, (42, 128), (74, 168), (0, 0, 0), -1)
    cv2.rectangle(frame, (220, 70), (236, 86), (0, 0, 0), -1)

    state = analyzer.analyze(frame, timestamp=1.0)

    assert state.nearest_obstacle is None


def test_detector_ignores_flat_ground_artifact_as_player():
    analyzer = DinoVisionAnalyzer(
        DinoVisionConfig(
            playfield_top_ratio=0.0,
            playfield_bottom_ratio=1.0,
            player_search_fraction=0.55,
            min_player_area=40,
            min_obstacle_area=40,
        )
    )

    frame = np.full((220, 420, 3), 255, dtype=np.uint8)
    cv2.line(frame, (0, 176), (419, 176), (0, 0, 0), 2)
    cv2.rectangle(frame, (42, 128), (74, 168), (0, 0, 0), -1)
    cv2.rectangle(frame, (42, 174), (240, 180), (0, 0, 0), -1)
    cv2.rectangle(frame, (260, 128), (290, 168), (0, 0, 0), -1)

    state = analyzer.analyze(frame, timestamp=1.0)

    assert state.player_box.h >= 30
    assert state.player_box.w < 100
    assert state.nearest_obstacle is not None


def test_detector_keeps_airborne_player_box():
    analyzer = DinoVisionAnalyzer(
        DinoVisionConfig(
            playfield_top_ratio=0.0,
            playfield_bottom_ratio=1.0,
            player_search_fraction=0.45,
            min_player_area=40,
            min_obstacle_area=40,
        )
    )

    frame = np.full((220, 420, 3), 255, dtype=np.uint8)
    cv2.line(frame, (0, 176), (419, 176), (0, 0, 0), 2)
    cv2.rectangle(frame, (42, 72), (74, 112), (0, 0, 0), -1)
    cv2.rectangle(frame, (220, 128), (250, 168), (0, 0, 0), -1)

    state = analyzer.analyze(frame, timestamp=1.0)

    assert state.player_box.y < 100
    assert state.player_box.bottom < state.ground_y - 20


def test_detector_rejects_obstacle_as_shifted_player():
    analyzer = DinoVisionAnalyzer(
        DinoVisionConfig(
            playfield_top_ratio=0.0,
            playfield_bottom_ratio=1.0,
            player_search_fraction=0.7,
            max_player_x_px=80,
            min_player_area=40,
            min_obstacle_area=40,
        )
    )

    first = np.full((220, 420, 3), 255, dtype=np.uint8)
    cv2.line(first, (0, 176), (419, 176), (0, 0, 0), 2)
    cv2.rectangle(first, (42, 128), (74, 168), (0, 0, 0), -1)
    analyzer.analyze(first, timestamp=1.0)

    shifted = np.full((220, 420, 3), 255, dtype=np.uint8)
    cv2.line(shifted, (0, 176), (419, 176), (0, 0, 0), 2)
    cv2.rectangle(shifted, (120, 128), (175, 168), (0, 0, 0), -1)

    state = analyzer.analyze(shifted, timestamp=1.1)

    assert abs(state.player_box.x - 42) <= 1


def test_detector_rejects_mid_height_player_artifact():
    analyzer = DinoVisionAnalyzer(
        DinoVisionConfig(
            playfield_top_ratio=0.0,
            playfield_bottom_ratio=1.0,
            player_search_fraction=0.7,
            min_player_area=40,
            min_obstacle_area=40,
        )
    )

    first = np.full((220, 420, 3), 255, dtype=np.uint8)
    cv2.line(first, (0, 176), (419, 176), (0, 0, 0), 2)
    cv2.rectangle(first, (42, 128), (74, 168), (0, 0, 0), -1)
    analyzer.analyze(first, timestamp=1.0)

    artifact = np.full((220, 420, 3), 255, dtype=np.uint8)
    cv2.line(artifact, (0, 176), (419, 176), (0, 0, 0), 2)
    cv2.rectangle(artifact, (42, 126), (130, 174), (0, 0, 0), -1)

    state = analyzer.analyze(artifact, timestamp=1.1)

    assert state.player_box.h >= 40
    assert state.player_box.w < 60


def test_detector_rejects_wide_standing_player_artifact():
    analyzer = DinoVisionAnalyzer(
        DinoVisionConfig(
            playfield_top_ratio=0.0,
            playfield_bottom_ratio=1.0,
            player_search_fraction=0.7,
            max_player_x_px=220,
            min_player_area=40,
            min_obstacle_area=40,
        )
    )

    first = np.full((260, 420, 3), 255, dtype=np.uint8)
    cv2.line(first, (0, 221), (419, 221), (0, 0, 0), 2)
    cv2.rectangle(first, (42, 156), (102, 213), (0, 0, 0), -1)
    analyzer.analyze(first, timestamp=1.0)

    artifact = np.full((260, 420, 3), 255, dtype=np.uint8)
    cv2.line(artifact, (0, 221), (419, 221), (0, 0, 0), 2)
    cv2.rectangle(artifact, (42, 150), (124, 220), (0, 0, 0), -1)

    state = analyzer.analyze(artifact, timestamp=1.1)

    assert state.player_box.w <= 70


def test_detector_rejects_implausible_vertical_player_jump():
    analyzer = DinoVisionAnalyzer(
        DinoVisionConfig(
            playfield_top_ratio=0.0,
            playfield_bottom_ratio=1.0,
            player_search_fraction=0.7,
            max_player_x_px=220,
            min_player_area=40,
            min_obstacle_area=40,
        )
    )

    airborne = np.full((260, 420, 3), 255, dtype=np.uint8)
    cv2.line(airborne, (0, 221), (419, 221), (0, 0, 0), 2)
    cv2.rectangle(airborne, (42, 45), (102, 108), (0, 0, 0), -1)
    analyzer.analyze(airborne, timestamp=1.0)

    artifact = np.full((260, 420, 3), 255, dtype=np.uint8)
    cv2.line(artifact, (0, 221), (419, 221), (0, 0, 0), 2)
    cv2.rectangle(artifact, (42, 156), (102, 220), (0, 0, 0), -1)

    state = analyzer.analyze(artifact, timestamp=1.1)

    assert state.player_box.y < 80
