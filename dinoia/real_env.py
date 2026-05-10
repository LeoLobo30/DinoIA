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
from .real_rl import RealActionGate, RealRLObservationBuilder
from .types import PolicyAction, VisionState
from .vision import DinoVisionAnalyzer


@dataclass(slots=True)
class RealEnvConfig:
    survival_reward: float = 0.02
    pass_reward: float = 0.35
    collision_penalty: float = -1.0
    action_penalty: float = -0.002
    blocked_action_penalty: float = -0.01
    unnecessary_jump_penalty: float = -0.03
    unnecessary_duck_penalty: float = -0.02
    missed_action_penalty: float = -0.12
    pass_detection_distance_px: float = 90.0
    safe_jump_distance_px: float = 140.0
    duck_bird_distance_px: float = 120.0
    step_interval_s: float = 1.0 / 30.0
    reset_wait_s: float = 0.8
    max_episode_steps: int = 3_000
    auto_focus: bool = True
    show_debug: bool = False
    enable_action_gate: bool = True


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
        self.action_gate = RealActionGate(min_action_interval_s=self.vision_config.min_action_interval_s)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self._last_state: VisionState | None = None
        self._episode_steps = 0
        self._passed_obstacles = 0
        self._last_reward_components: dict[str, float] = {}
        self._last_requested_action = PolicyAction.NOOP
        self._last_executed_action = PolicyAction.NOOP
        self._last_step_at = 0.0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._episode_steps = 0
        self._passed_obstacles = 0
        self._last_reward_components = {}
        self._last_requested_action = PolicyAction.NOOP
        self._last_executed_action = PolicyAction.NOOP
        self.action_gate.last_jump_at = 0.0
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
        previous_state = self._last_state
        requested_action = self._map_action(int(action))
        executed_action = self._filter_action(requested_action, previous_state)
        self._last_requested_action = requested_action
        self._last_executed_action = executed_action
        self.controller.perform(executed_action)
        state = self._capture_state()
        self._last_state = state

        terminated = bool(state.game_over)
        truncated = self._episode_steps >= self.env_config.max_episode_steps
        reward_components = self._reward_components(
            requested_action,
            executed_action,
            previous_state,
            state,
            terminated,
        )
        reward = sum(reward_components.values())
        if terminated:
            reward += self.env_config.collision_penalty
            reward_components["collision"] = self.env_config.collision_penalty

        if reward_components.get("pass", 0.0) > 0:
            self._passed_obstacles += 1
        self._last_reward_components = reward_components

        observation = self.observation_builder.build(state)
        return observation, float(reward), terminated, truncated, self._info(state)

    def render(self):
        if self._last_state is None:
            return None
        return self.analyzer.draw_overlay(self._last_state)

    def close(self):
        self.controller.release_duck()
        close = getattr(self.capture, "close", None)
        if callable(close):
            close()

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

    def _filter_action(self, action: PolicyAction, state: VisionState | None) -> PolicyAction:
        if not self.env_config.enable_action_gate:
            return action
        if state is None:
            return action
        grounded = state.player_box.bottom >= state.ground_y - 4
        return self.action_gate.filter(action, grounded=grounded, now=time.time())

    def _reward_components(
        self,
        requested_action: PolicyAction,
        executed_action: PolicyAction,
        previous_state: VisionState | None,
        state: VisionState,
        terminated: bool,
    ) -> dict[str, float]:
        components = {"survival": self.env_config.survival_reward}
        if executed_action is not PolicyAction.NOOP:
            components["action"] = self.env_config.action_penalty
        if requested_action is not executed_action:
            components["blocked_action"] = self.env_config.blocked_action_penalty

        if self._passed_obstacle(previous_state, state, terminated):
            components["pass"] = self.env_config.pass_reward

        bad_action_penalty = self._bad_action_penalty(requested_action, previous_state or state)
        if bad_action_penalty:
            components["bad_action"] = bad_action_penalty
        missed_action_penalty = self._missed_action_penalty(requested_action, previous_state or state)
        if missed_action_penalty:
            components["missed_action"] = missed_action_penalty

        return components

    def _passed_obstacle(
        self,
        previous_state: VisionState | None,
        state: VisionState,
        terminated: bool,
    ) -> bool:
        if previous_state is None or terminated:
            return False
        previous_obstacle = previous_state.nearest_obstacle
        previous_distance = previous_state.nearest_distance_px
        if previous_obstacle is None or previous_distance is None:
            return False
        if previous_distance > self.env_config.pass_detection_distance_px:
            return False

        current_distance = state.nearest_distance_px
        if state.nearest_obstacle is None or current_distance is None:
            return True

        # A large forward distance jump usually means the close obstacle left
        # the playfield and the detector is now seeing the next one.
        return current_distance > previous_distance + self.env_config.pass_detection_distance_px

    def _bad_action_penalty(self, action: PolicyAction, state: VisionState) -> float:
        nearest = state.nearest_obstacle
        distance = state.nearest_distance_px
        grounded = state.player_box.bottom >= state.ground_y - 4

        if action is PolicyAction.JUMP:
            if not grounded:
                return self.env_config.unnecessary_jump_penalty
            if nearest is None or distance is None or distance > self.env_config.safe_jump_distance_px:
                return self.env_config.unnecessary_jump_penalty

        if action is PolicyAction.DUCK:
            bird_is_close = (
                nearest is not None
                and nearest.label == "bird"
                and distance is not None
                and distance <= self.env_config.duck_bird_distance_px
            )
            if not grounded or not bird_is_close:
                return self.env_config.unnecessary_duck_penalty

        return 0.0

    def _missed_action_penalty(self, action: PolicyAction, state: VisionState) -> float:
        if action is not PolicyAction.NOOP:
            return 0.0

        nearest = state.nearest_obstacle
        distance = state.nearest_distance_px
        if nearest is None or distance is None:
            return 0.0

        if nearest.label == "bird" and distance <= self.env_config.duck_bird_distance_px:
            return self.env_config.missed_action_penalty
        if nearest.label != "bird" and distance <= self._jump_trigger_distance(state):
            return self.env_config.missed_action_penalty
        return 0.0

    def _jump_trigger_distance(self, state: VisionState) -> float:
        return self.vision_config.jump_base_distance_px + min(
            state.estimated_speed_px_s * self.vision_config.jump_speed_factor,
            self.vision_config.jump_base_distance_px * 2.2,
        )

    def _info(self, state: VisionState) -> dict[str, Any]:
        return {
            "game_over": state.game_over,
            "speed": state.estimated_speed_px_s,
            "distance": state.nearest_distance_px,
            "obstacle_count": len(state.obstacles),
            "passed_obstacles": self._passed_obstacles,
            "reward_components": dict(self._last_reward_components),
            "requested_action": self._last_requested_action.value,
            "executed_action": self._last_executed_action.value,
            "action_blocked": self._last_requested_action is not self._last_executed_action,
        }
