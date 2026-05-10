from __future__ import annotations

import numpy as np

from dinoia.config import RealGameConfig
from dinoia.real_env import DinoRealEnv, RealEnvConfig
from dinoia.types import ObstacleDetection, PolicyAction, Rect, VisionState


class FakeCapture:
    def __init__(self):
        self.focuses = 0

    def focus_window(self):
        self.focuses += 1

    def capture(self):
        return np.zeros((200, 400, 3), dtype=np.uint8)


class FakeAnalyzer:
    def __init__(self, states=None):
        self.game_over = False
        self.states = list(states or [])

    def analyze(self, frame, timestamp=None):
        if self.states:
            state = self.states.pop(0)
            return VisionState(
                timestamp=timestamp or state.timestamp,
                frame=frame,
                playfield=frame,
                player_box=state.player_box,
                ground_y=state.ground_y,
                obstacles=state.obstacles,
                nearest_obstacle=state.nearest_obstacle,
                nearest_distance_px=state.nearest_distance_px,
                estimated_speed_px_s=state.estimated_speed_px_s,
                estimated_player_vy_px_s=state.estimated_player_vy_px_s,
                game_over=state.game_over,
            )
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


def _state_with_obstacle(distance=None, label="cactus", game_over=False, player_y=120):
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    player = Rect(40, player_y, 30, 44)
    obstacle = None
    obstacles = []
    if distance is not None:
        obstacle = ObstacleDetection(
            rect=Rect(player.right + int(distance), 122, 24, 40),
            label=label,
            distance_px=float(distance),
        )
        obstacles = [obstacle]
    return VisionState(
        timestamp=1.0,
        frame=frame,
        playfield=frame,
        player_box=player,
        ground_y=164,
        obstacles=obstacles,
        nearest_obstacle=obstacle,
        nearest_distance_px=float(distance) if distance is not None else None,
        estimated_speed_px_s=320.0,
        estimated_player_vy_px_s=0.0,
        game_over=game_over,
    )


def test_real_env_reset_and_step_without_keyboard():
    controller = FakeController()
    env = DinoRealEnv(
        real_config=RealGameConfig(focus_window=False),
        env_config=RealEnvConfig(step_interval_s=0.0, reset_wait_s=0.0),
        capture=FakeCapture(),
        analyzer=FakeAnalyzer(states=[_state_with_obstacle(60), _state_with_obstacle(45)]),
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


def test_real_env_rewards_passing_close_obstacle():
    env = DinoRealEnv(
        real_config=RealGameConfig(focus_window=False),
        env_config=RealEnvConfig(step_interval_s=0.0, reset_wait_s=0.0),
        capture=FakeCapture(),
        analyzer=FakeAnalyzer(states=[_state_with_obstacle(40), _state_with_obstacle(None)]),
        controller=FakeController(),
    )

    env.reset()
    _, reward, terminated, _, info = env.step(0)

    assert terminated is False
    assert reward > env.env_config.survival_reward
    assert info["passed_obstacles"] == 1
    assert info["reward_components"]["pass"] == env.env_config.pass_reward


def test_real_env_penalizes_unnecessary_jump():
    env = DinoRealEnv(
        real_config=RealGameConfig(focus_window=False),
        env_config=RealEnvConfig(step_interval_s=0.0, reset_wait_s=0.0),
        capture=FakeCapture(),
        analyzer=FakeAnalyzer(states=[_state_with_obstacle(None), _state_with_obstacle(None)]),
        controller=FakeController(),
    )

    env.reset()
    _, reward, terminated, _, info = env.step(1)

    assert terminated is False
    assert reward < env.env_config.survival_reward
    assert info["reward_components"]["bad_action"] == env.env_config.unnecessary_jump_penalty


def test_real_env_penalizes_noop_when_obstacle_requires_action():
    env = DinoRealEnv(
        real_config=RealGameConfig(focus_window=False),
        env_config=RealEnvConfig(step_interval_s=0.0, reset_wait_s=0.0),
        capture=FakeCapture(),
        analyzer=FakeAnalyzer(states=[_state_with_obstacle(35), _state_with_obstacle(20)]),
        controller=FakeController(),
    )

    env.reset()
    _, reward, terminated, _, info = env.step(0)

    assert terminated is False
    assert reward < env.env_config.survival_reward
    assert info["reward_components"]["missed_action"] == env.env_config.missed_action_penalty


def test_real_env_blocks_airborne_jump_like_runner():
    env = DinoRealEnv(
        real_config=RealGameConfig(focus_window=False),
        env_config=RealEnvConfig(step_interval_s=0.0, reset_wait_s=0.0),
        capture=FakeCapture(),
        analyzer=FakeAnalyzer(states=[_state_with_obstacle(60, player_y=70), _state_with_obstacle(45, player_y=75)]),
        controller=FakeController(),
    )

    env.reset()
    _, reward, terminated, _, info = env.step(1)

    assert terminated is False
    assert info["requested_action"] == PolicyAction.JUMP.value
    assert info["executed_action"] == PolicyAction.NOOP.value
    assert info["action_blocked"] is True
    assert info["reward_components"]["blocked_action"] == env.env_config.blocked_action_penalty
    assert reward < env.env_config.survival_reward
