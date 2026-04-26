from __future__ import annotations

import numpy as np

from dinoia.config import DinoVisionConfig
from dinoia.decision import DinoRulePolicy
from dinoia.types import ObstacleDetection, PolicyAction, Rect, VisionState


def _vision_state(label: str, distance: float, speed: float = 0.0) -> VisionState:
    obstacle = ObstacleDetection(rect=Rect(150, 120, 24, 40), label=label, distance_px=distance, speed_px_s=speed)
    return VisionState(
        timestamp=1.0,
        frame=np.zeros((200, 400, 3), dtype=np.uint8),
        playfield=np.zeros((180, 400, 3), dtype=np.uint8),
        player_box=Rect(40, 120, 30, 44),
        ground_y=160,
        obstacles=[obstacle],
        nearest_obstacle=obstacle,
        nearest_distance_px=distance,
        estimated_speed_px_s=speed,
        game_over=False,
    )


def test_policy_jumps_for_close_cactus_and_ducks_for_bird():
    policy = DinoRulePolicy(DinoVisionConfig(min_action_interval_s=0.0))

    jump_state = _vision_state("cactus", distance=20.0, speed=300.0)
    duck_state = _vision_state("bird", distance=20.0, speed=300.0)

    assert policy.decide(jump_state, now=10.0).action == PolicyAction.JUMP
    assert policy.decide(duck_state, now=20.0).action == PolicyAction.DUCK
