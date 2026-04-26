from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .capture import DinoScreenCapture
from .config import DinoVisionConfig, RealGameConfig, SimConfig
from .control import KeyboardController
from .observations import OBSERVATION_SIZE
from .real_rl import RealRLObservationBuilder
from .types import PolicyAction, VisionState
from .vision import DinoVisionAnalyzer


@dataclass(slots=True)
class RealEnvConfig:
    survival_reward: float = 0.02
    collision_penalty: float = -1.0
    step_interval_s: float = 1.0 / 30.0
    reset_wait_s: float = 0.8
    max_episode_steps: int = 3_000
    auto_focus: bool = True
    show_debug: bool = False


class DinoRealEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        *,
        real_config: RealGameConfig | None = None,
        vision_config: DinoVisionConfig | None = None,
        sim_config: SimConfig | None = None,
        env_config: RealEnvConfig | None = None,
        capture: DinoScreenCapture | None = None,
        analyzer: DinoVisionAnalyzer | None = None,
        controller: KeyboardController | None = None,
    ) -> None:
        super().__init__()
        self.real_config = real_config or RealGameConfig(debug=False)
        self.vision_config = vision_config or DinoVisionConfig()
        self.env_config = env_config or RealEnvConfig()
        self.capture = capture or DinoScreenCapture(self.real_config)
        self.analyzer = analyzer or DinoVisionAnalyzer(self.vision_config)
        self.controller = controller or KeyboardController()
        self.observation_builder = RealRLObservationBuilder(sim_config=sim_config)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self._last_state: VisionState | None = None
        self._episode_steps = 0
        self._last_step_at = 0.0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._episode_steps = 0
        self.controller.release_duck()
        if self.env_config.auto_focus and self.real_config.focus_window:
            self._focus_window()
        self.controller.restart()
        time.sleep(self.env_config.reset_wait_s)
        state = self._capture_state()
        self._last_state = state
        self._last_step_at = time.time()
        return self.observation_builder.build(state), self._info(state)

    def step(self, action: int):
        self._episode_steps += 1
        self._pace()
        self.controller.perform(self._map_action(int(action)))
        state = self._capture_state()
        self._last_state = state

        terminated = bool(state.game_over)
        truncated = self._episode_steps >= self.env_config.max_episode_steps
        reward = self.env_config.survival_reward
        if terminated:
            reward += self.env_config.collision_penalty

        observation = self.observation_builder.build(state)
        return observation, float(reward), terminated, truncated, self._info(state)

    def render(self):
        if self._last_state is None:
            return None
        return self.analyzer.draw_overlay(self._last_state)

    def close(self):
        self.controller.release_duck()

    def _capture_state(self) -> VisionState:
        frame = self.capture.capture()
        return self.analyzer.analyze(frame, timestamp=time.time())

    def _pace(self) -> None:
        elapsed = time.time() - self._last_step_at
        sleep_for = max(0.0, self.env_config.step_interval_s - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last_step_at = time.time()

    def _focus_window(self) -> None:
        try:
            self.capture.focus_window()
        except Exception:
            pass

    @staticmethod
    def _map_action(action: int) -> PolicyAction:
        if action == 1:
            return PolicyAction.JUMP
        if action == 2:
            return PolicyAction.DUCK
        return PolicyAction.NOOP

    def _info(self, state: VisionState) -> dict[str, Any]:
        return {
            "game_over": state.game_over,
            "speed": state.estimated_speed_px_s,
            "distance": state.nearest_distance_px,
            "obstacle_count": len(state.obstacles),
        }
