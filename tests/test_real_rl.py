from __future__ import annotations

import numpy as np

from dinoia.real_rl import RealActionGate, RealRLObservationBuilder
from dinoia.types import ObstacleDetection, PolicyAction, Rect, VisionState


def test_real_rl_observation_matches_sim_shape():
    obstacle = ObstacleDetection(rect=Rect(140, 120, 24, 40), label="cactus", distance_px=36.0)
    state = VisionState(
        timestamp=1.0,
        frame=np.zeros((200, 400, 3), dtype=np.uint8),
        playfield=np.zeros((180, 400, 3), dtype=np.uint8),
        player_box=Rect(40, 120, 30, 44),
        ground_y=160,
        obstacles=[obstacle],
        nearest_obstacle=obstacle,
        nearest_distance_px=36.0,
        estimated_speed_px_s=320.0,
        estimated_player_vy_px_s=-180.0,
        game_over=False,
    )

    obs = RealRLObservationBuilder().build(state)

    assert obs.shape == (13,)
    assert np.isfinite(obs).all()
    assert obs[4] > 0.45
    assert np.isclose(obs[5], 0.09, atol=0.01)


def test_real_rl_observation_uses_speed_floor_when_obstacle_exists():
    obstacle = ObstacleDetection(rect=Rect(220, 120, 24, 40), label="cactus", distance_px=150.0)
    state = VisionState(
        timestamp=1.0,
        frame=np.zeros((200, 400, 3), dtype=np.uint8),
        playfield=np.zeros((180, 400, 3), dtype=np.uint8),
        player_box=Rect(40, 120, 30, 44),
        ground_y=160,
        obstacles=[obstacle],
        nearest_obstacle=obstacle,
        nearest_distance_px=150.0,
        estimated_speed_px_s=0.0,
        estimated_player_vy_px_s=0.0,
        game_over=False,
    )

    obs = RealRLObservationBuilder().build(state)

    assert obs[4] > 0.3


def test_real_rl_observation_does_not_duck_when_airborne():
    state = VisionState(
        timestamp=1.0,
        frame=np.zeros((200, 400, 3), dtype=np.uint8),
        playfield=np.zeros((180, 400, 3), dtype=np.uint8),
        player_box=Rect(40, 112, 62, 38),
        ground_y=156,
        obstacles=[],
        nearest_obstacle=None,
        nearest_distance_px=None,
        estimated_speed_px_s=320.0,
        estimated_player_vy_px_s=0.0,
        game_over=False,
    )

    obs = RealRLObservationBuilder().build(state)

    assert obs[2] == 0.0
    assert obs[3] == 0.0


def test_real_action_gate_blocks_spam_and_airborne_actions():
    gate = RealActionGate(min_action_interval_s=0.5)

    assert gate.filter(PolicyAction.JUMP, grounded=False, now=1.0) == PolicyAction.NOOP
    assert gate.filter(PolicyAction.JUMP, grounded=True, now=1.0) == PolicyAction.JUMP
    assert gate.filter(PolicyAction.JUMP, grounded=True, now=1.2) == PolicyAction.NOOP
    assert gate.filter(PolicyAction.JUMP, grounded=True, now=1.7) == PolicyAction.JUMP

