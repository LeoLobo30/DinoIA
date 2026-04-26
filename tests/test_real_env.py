from __future__ import annotations

import numpy as np

from dinoia.config import RealGameConfig
from dinoia.real_env import DinoRealEnv, RealEnvConfig
from dinoia.types import PolicyAction, Rect, VisionState


class FakeCapture:
    def __init__(self):
        self.focuses = 0

    def focus_window(self):
        self.focuses += 1

    def capture(self):
        return np.zeros((200, 400, 3), dtype=np.uint8)


class FakeAnalyzer:
    def __init__(self):
        self.game_over = False

    def analyze(self, frame, timestamp=None):
        return VisionState(
            timestamp=timestamp or 1.0,
            frame=frame,
            playfield=frame,
            player_box=Rect(40, 120, 30, 44),
            ground_y=164,
            obstacles=[],
            nearest_obstacle=None,
            nearest_distance_px=None,
            estimated_speed_px_s=320.0,
            estimated_player_vy_px_s=0.0,
            game_over=self.game_over,
        )

    def draw_overlay(self, state):
        return state.frame


class FakeController:
    def __init__(self):
        self.actions = []

    def release_duck(self):
        self.actions.append(PolicyAction.NOOP)

    def restart(self):
        self.actions.append(PolicyAction.JUMP)

    def perform(self, action):
        self.actions.append(action)


def test_real_env_reset_and_step_without_keyboard():
    controller = FakeController()
    env = DinoRealEnv(
        real_config=RealGameConfig(focus_window=False),
        env_config=RealEnvConfig(step_interval_s=0.0, reset_wait_s=0.0),
        capture=FakeCapture(),
        analyzer=FakeAnalyzer(),
        controller=controller,
    )

    obs, info = env.reset()
    next_obs, reward, terminated, truncated, step_info = env.step(1)

    assert obs.shape == (13,)
    assert next_obs.shape == (13,)
    assert reward > 0
    assert terminated is False
    assert truncated is False
    assert info["game_over"] is False
    assert step_info["game_over"] is False
    assert controller.actions[-1] is PolicyAction.JUMP


def test_real_env_terminates_on_game_over():
    analyzer = FakeAnalyzer()
    analyzer.game_over = True
    env = DinoRealEnv(
        real_config=RealGameConfig(focus_window=False),
        env_config=RealEnvConfig(step_interval_s=0.0, reset_wait_s=0.0),
        capture=FakeCapture(),
        analyzer=analyzer,
        controller=FakeController(),
    )

    env.reset()
    _, reward, terminated, _, info = env.step(0)

    assert terminated is True
    assert reward < 0
    assert info["game_over"] is True
