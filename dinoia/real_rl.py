from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import DQN
import torch

from .config import DinoVisionConfig, RealGameConfig, SimConfig
from .control import KeyboardController
from .observations import build_dino_observation
from .real_game import DinoRealGameRunner, RunnerStats
from .types import PolicyAction, Rect, VisionState


@dataclass(slots=True)
class RealRLResult:
    stats: RunnerStats
    steps: int
    model_path: str


class RealRLObservationBuilder:
    def __init__(self, sim_config: SimConfig | None = None):
        self.sim_config = sim_config or SimConfig()

    def build(self, state: VisionState):
        playfield = state.playfield
        height, width = playfield.shape[:2]
        player = state.player_box
        nearest = state.nearest_obstacle
        sorted_obstacles = sorted(state.obstacles, key=lambda obstacle: obstacle.distance_px)
        second = sorted_obstacles[1] if len(sorted_obstacles) > 1 else None

        x_scale = self._x_scale(width)
        y_scale = self._y_scale(height)
        scaled_player = self._scale_rect(player, x_scale=x_scale, y_scale=y_scale)
        scaled_nearest = self._scale_rect(nearest.rect, x_scale=x_scale, y_scale=y_scale) if nearest is not None else None
        scaled_second = self._scale_rect(second.rect, x_scale=x_scale, y_scale=y_scale) if second is not None else None
        floor_y = self._scale_y(state.ground_y, y_scale)
        nearest_distance = self._scale_x(nearest.distance_px, x_scale) if nearest is not None else None
        second_distance = self._scale_x(second.distance_px, x_scale) if second is not None else None
        current_speed = self._estimate_current_speed(state, x_scale)
        player_vy = self._scale_y(state.estimated_player_vy_px_s, y_scale)
        player_bottom = self._scale_y(player.bottom, y_scale)
        duck_height = self._scale_y(player.h, y_scale)
        grounded = player.bottom >= state.ground_y - 4

        observation = build_dino_observation(
            player_y_px=scaled_player.y,
            player_vy_px=player_vy,
            grounded=grounded,
            ducking=grounded and duck_height <= float(self.sim_config.duck_height) + 2.0,
            current_speed_px_s=current_speed,
            screen_width_px=float(self.sim_config.screen_width),
            screen_height_px=float(self.sim_config.screen_height),
            floor_y_px=float(self.sim_config.screen_height - self.sim_config.ground_height),
            max_speed_px_s=self.sim_config.max_speed,
            nearest_distance_px=nearest_distance,
            nearest_width_px=float(scaled_nearest.w) if scaled_nearest is not None else None,
            nearest_height_px=float(scaled_nearest.h) if scaled_nearest is not None else None,
            nearest_is_bird=nearest.label == "bird" if nearest is not None else False,
            second_distance_px=second_distance,
            second_width_px=float(scaled_second.w) if scaled_second is not None else None,
            second_height_px=float(scaled_second.h) if scaled_second is not None else None,
            obstacle_count=len(sorted_obstacles),
        )
        return observation

    def _x_scale(self, width: int) -> float:
        return float(self.sim_config.screen_width / max(1.0, float(width)))

    def _y_scale(self, height: int) -> float:
        return float(self.sim_config.screen_height / max(1.0, float(height)))

    @staticmethod
    def _scale_x(value: float, scale: float) -> float:
        return float(value * scale)

    @staticmethod
    def _scale_y(value: float, scale: float) -> float:
        return float(value * scale)

    def _estimate_current_speed(self, state: VisionState, x_scale: float) -> float:
        scaled_speed = max(0.0, self._scale_x(state.estimated_speed_px_s, x_scale))
        if state.nearest_obstacle is None:
            return scaled_speed
        if scaled_speed >= self.sim_config.base_speed * 0.35:
            return scaled_speed
        return float(self.sim_config.base_speed)

    @staticmethod
    def _scale_rect(rect: Rect, *, x_scale: float, y_scale: float) -> Rect:
        return Rect(
            x=int(round(rect.x * x_scale)),
            y=int(round(rect.y * y_scale)),
            w=max(1, int(round(rect.w * x_scale))),
            h=max(1, int(round(rect.h * y_scale))),
        )


@dataclass(slots=True)
class RealActionGate:
    min_action_interval_s: float
    last_jump_at: float = 0.0

    def filter(self, action: PolicyAction, *, grounded: bool, now: float) -> PolicyAction:
        if action is PolicyAction.JUMP:
            if not grounded:
                return PolicyAction.NOOP
            if now - self.last_jump_at < self.min_action_interval_s:
                return PolicyAction.NOOP
            self.last_jump_at = now
            return action
        if action is PolicyAction.DUCK and not grounded:
            return PolicyAction.NOOP
        return action


class DinoRealRLRunner:
    def __init__(
        self,
        model_path: str | Path,
        real_config: RealGameConfig | None = None,
        vision_config: DinoVisionConfig | None = None,
        sim_config: SimConfig | None = None,
        device: str = "auto",
        debug: bool | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.runner = DinoRealGameRunner(real_config=real_config, vision_config=vision_config, debug=debug)
        self.observation_builder = RealRLObservationBuilder(sim_config=sim_config)
        resolved_device = device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        if resolved_device == "cuda" and not torch.cuda.is_available():
            resolved_device = "cpu"
        self.model = DQN.load(self.model_path, device=resolved_device)
        self.device = resolved_device
        self.controller = KeyboardController()
        self.action_gate = RealActionGate(min_action_interval_s=self.runner.vision_config.min_action_interval_s)
        self.step_count = 0

    def run(self, duration_s: float | None = None) -> RealRLResult:
        start = time.time()
        frame_interval = 1.0 / max(1.0, self.runner.real_config.frame_rate)
        self.runner.capture.focus_window() if self.runner.real_config.focus_window else None

        try:
            if self.runner.real_config.auto_start:
                time.sleep(0.5)
                self.controller.jump()

            while True:
                now = time.time()
                if duration_s is not None and now - start >= duration_s:
                    break

                if self.runner.real_config.focus_window and self.runner.real_config.keep_window_foreground:
                    try:
                        self.runner.capture.focus_window()
                    except Exception:
                        pass

                frame = self.runner.capture.capture()
                state = self.runner.analyzer.analyze(frame, timestamp=now)

                if self.runner.real_config.auto_restart and state.game_over:
                    self.controller.release_duck()
                    self.controller.restart()
                    time.sleep(0.6)
                    continue

                obs = self.observation_builder.build(state)
                action, _ = self.model.predict(obs, deterministic=True)
                grounded = state.player_box.bottom >= state.ground_y - 4
                filtered_action = self.action_gate.filter(self._map_action(int(action)), grounded=grounded, now=now)
                executed = self.controller.perform(filtered_action)
                self.runner._count_action(executed.action)
                self.step_count += 1

                if self.runner.real_config.debug:
                    overlay = self.runner.analyzer.draw_overlay(state)
                    import cv2

                    cv2.imshow("DinoIA - Real RL", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                self.runner.stats.frames += 1
                sleep_for = max(0.0, frame_interval - (time.time() - now))
                time.sleep(sleep_for)
        finally:
            self.controller.release_duck()
            if self.runner.real_config.debug:
                import cv2

                cv2.destroyAllWindows()

        return RealRLResult(stats=self.runner.stats, steps=self.step_count, model_path=self.model_path)

    @staticmethod
    def _map_action(action: int):
        if action == 1:
            return PolicyAction.JUMP
        if action == 2:
            return PolicyAction.DUCK
        return PolicyAction.NOOP
