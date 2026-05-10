from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import DinoVisionConfig, SimConfig
from ..decision import DinoRulePolicy
from ..observations import OBSERVATION_SIZE, build_dino_observation
from ..types import ObstacleDetection, PolicyAction, Rect, VisionState


@dataclass(slots=True)
class SimObstacle:
    kind: str
    x: float
    y: float
    w: int
    h: int
    passed: bool = False

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.h


class DinoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, config: SimConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or SimConfig()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)

        self.floor_y = self.config.screen_height - self.config.ground_height
        self.player_y = float(self.floor_y - self.config.stand_height)
        self.player_vy = 0.0
        self.grounded = True
        self.ducking = False
        self.score = 0.0
        self.elapsed_steps = 0
        self.passed_obstacles = 0
        self._spawn_timer = 0.0
        self._obstacles: list[SimObstacle] = []
        self._episode_return = 0.0
        self._last_render: np.ndarray | None = None
        self._episode_base_speed = float(self.config.base_speed)
        self._episode_speed_acceleration = float(self.config.speed_acceleration)
        self._episode_max_speed = float(self.config.max_speed)
        self._episode_gravity = float(self.config.gravity)
        self._episode_jump_velocity = float(self.config.jump_velocity)
        self._episode_bird_probability = float(self.config.bird_probability)
        self._episode_min_spawn_gap_px = int(self.config.min_spawn_gap_px)
        self._episode_max_spawn_gap_px = int(self.config.max_spawn_gap_px)
        self._episode_obstacle_scale = 1.0
        self._episode_observation_noise = float(self.config.observation_noise)
        self._action_buffer: list[int] = []
        self._observation_buffer: list[np.ndarray] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.floor_y = self.config.screen_height - self.config.ground_height
        self.player_y = float(self.floor_y - self.config.stand_height)
        self.player_vy = 0.0
        self.grounded = True
        self.ducking = False
        self.score = 0.0
        self.elapsed_steps = 0
        self.passed_obstacles = 0
        self._episode_return = 0.0
        self._episode_elapsed_time = 0.0
        self._obstacles = []
        self._sample_episode_parameters()
        self._spawn_timer = max(float(self.config.initial_obstacle_delay_s), self._next_spawn_delay())
        observation = self._get_observation()
        self._reset_latency_buffers(observation)
        info = {"speed": self.current_speed, "passed_obstacles": self.passed_obstacles}
        return observation, info

    def step(self, action: int):
        self.elapsed_steps += 1
        policy_dt = float(self.config.step_interval_s)
        physics_dt = float(self.config.physics_step_interval_s)
        substeps = max(1, int(round(policy_dt / max(1e-6, physics_dt))))
        requested_action = int(action)
        executed_action = self._delay_action(requested_action)
        self._apply_action(executed_action)

        collision = False
        passed_now = False
        for _ in range(substeps):
            self._advance_player(physics_dt)
            self._advance_obstacles(physics_dt)
            self._maybe_spawn_obstacle(physics_dt)
            self._episode_elapsed_time += physics_dt
            passed_now = self._consume_pass_rewards() or passed_now
            collision = self._check_collision()
            if collision:
                break

        reward = self.config.survival_reward
        if passed_now:
            reward += self.config.pass_reward
        reward += self._action_shaping_reward(requested_action)

        terminated = False
        if collision:
            terminated = True
            reward += self.config.collision_penalty

        truncated = self.elapsed_steps >= self.config.max_episode_steps
        self.score += reward
        self._episode_return += reward
        observation = self._get_observation()
        info = {
            "speed": self.current_speed,
            "score": self.score,
            "passed_obstacles": self.passed_obstacles,
            "collision": collision,
            "episode_return": self._episode_return,
            "requested_action": requested_action,
            "executed_action": executed_action,
        }
        return observation, reward, terminated, truncated, info

    def render(self):
        frame = self._render_frame()
        self._last_render = frame
        if self.render_mode == "human":
            cv2.imshow("DinoSim", frame)
            cv2.waitKey(1)
        return frame

    def close(self):
        if self.render_mode == "human":
            cv2.destroyAllWindows()

    @property
    def current_speed(self) -> float:
        speed = self._episode_base_speed + self._episode_elapsed_time * self._episode_speed_acceleration
        return float(min(self._episode_max_speed, speed))

    def spawn_obstacle(
        self,
        kind: str = "cactus",
        *,
        x: float | None = None,
        width: int | None = None,
        height: int | None = None,
        y: float | None = None,
    ) -> SimObstacle:
        x = float(self.config.screen_width + 15 if x is None else x)
        if kind == "bird":
            width = width or self._sample_scaled_dimension(28, 38, self._episode_obstacle_scale)
            height = height or self._sample_scaled_dimension(18, 28, self._episode_obstacle_scale)
            y = float(
                y
                if y is not None
                else self.floor_y - self._sample_scaled_dimension(56, 82, self._episode_obstacle_scale)
            )
        else:
            width = width or self._sample_scaled_dimension(18, 36, self._episode_obstacle_scale)
            height = height or self._sample_scaled_dimension(34, 58, self._episode_obstacle_scale)
            y = float(y if y is not None else self.floor_y - height)

        obstacle = SimObstacle(kind=kind, x=x, y=y, w=width, h=height)
        self._obstacles.append(obstacle)
        return obstacle

    def _apply_action(self, action: int) -> None:
        if action == 1 and self.grounded:
            self.player_vy = -self._episode_jump_velocity
            self.grounded = False
            self.ducking = False
            return

        if self.grounded:
            if action == 2:
                self.ducking = True
            elif action == 0:
                self.ducking = False

    def _advance_player(self, dt: float) -> None:
        if not self.grounded:
            self.player_vy += self._episode_gravity * dt
            self.player_y += self.player_vy * dt
            if self.player_y >= self.floor_y - self.config.stand_height:
                self.player_y = float(self.floor_y - self.config.stand_height)
                self.player_vy = 0.0
                self.grounded = True
                self.ducking = False
        else:
            height = self.player_height
            self.player_y = float(self.floor_y - height)

    def _advance_obstacles(self, dt: float) -> None:
        speed = self.current_speed
        for obstacle in self._obstacles:
            obstacle.x -= speed * dt

    def _maybe_spawn_obstacle(self, dt: float) -> None:
        self._spawn_timer -= dt
        if self._spawn_timer > 0:
            return
        kind = "bird" if self.np_random.random() < self._episode_bird_probability and self.current_speed > 360 else "cactus"
        self.spawn_obstacle(kind=kind)
        self._spawn_timer = self._next_spawn_delay()

    def _sample_spawn_gap(self) -> float:
        min_gap = self._episode_min_spawn_gap_px
        max_gap = self._episode_max_spawn_gap_px
        if max_gap <= min_gap:
            max_gap = min_gap + 20
        return float(self.np_random.integers(min_gap, max_gap + 1))

    def _next_spawn_delay(self) -> float:
        gap_px = self._sample_spawn_gap() * float(self.config.gap_coefficient)
        return gap_px / max(1.0, self.current_speed)

    def _check_collision(self) -> bool:
        player_boxes = self._player_collision_boxes()
        for obstacle in self._obstacles:
            obstacle_boxes = self._obstacle_collision_boxes(obstacle)
            for player_box in player_boxes:
                for obstacle_box in obstacle_boxes:
                    if self._rects_intersect(player_box, obstacle_box):
                        return True
        return False

    def _consume_pass_rewards(self) -> bool:
        passed_now = False
        player = self._player_rect()
        remaining: list[SimObstacle] = []
        for obstacle in self._obstacles:
            if not obstacle.passed and obstacle.right < player[0]:
                obstacle.passed = True
                self.passed_obstacles += 1
                passed_now = True
            if obstacle.right > -40:
                remaining.append(obstacle)
        self._obstacles = remaining
        return passed_now

    def _reset_latency_buffers(self, observation: np.ndarray) -> None:
        action_latency = max(0, int(self.config.action_latency_steps))
        observation_latency = max(0, int(self.config.observation_latency_steps))
        self._action_buffer = [0] * action_latency
        self._observation_buffer = [observation.copy() for _ in range(observation_latency)]

    def _delay_action(self, action: int) -> int:
        if not self._action_buffer:
            return action

        delayed_action = self._action_buffer.pop(0)
        self._action_buffer.append(action)
        return delayed_action

    def _player_rect(self) -> tuple[float, float, float, float]:
        height = self.player_height
        return float(self.config.player_x), float(self.player_y), float(self.config.player_width), float(height)

    @property
    def player_height(self) -> int:
        return self.config.duck_height if self.ducking and self.grounded else self.config.stand_height

    def _player_collision_boxes(self) -> list[Rect]:
        px, py, _, _ = self._player_rect()
        if self.ducking and self.grounded:
            return [Rect(int(px + 1), int(py + 18), 55, 25)]
        return [Rect(int(px + 22), int(py + 0), 17, 16)]

    @staticmethod
    def _rects_intersect(a: Rect, b: Rect) -> bool:
        return not (
            a.right <= b.x
            or b.right <= a.x
            or a.bottom <= b.y
            or b.bottom <= a.y
        )

    def _obstacle_collision_boxes(self, obstacle: SimObstacle) -> list[Rect]:
        rect = Rect(int(obstacle.left), int(obstacle.top), int(obstacle.w), int(obstacle.h))
        if obstacle.kind == "bird":
            pad_x = max(1, rect.w // 10)
            pad_y = max(1, rect.h // 5)
            return [Rect(rect.x + pad_x, rect.y + pad_y, max(1, rect.w - pad_x * 2), max(1, rect.h - pad_y * 2))]

        inset_x = max(1, rect.w // 8)
        inset_top = max(1, rect.h // 12)
        inset_bottom = max(1, rect.h // 10)
        return [Rect(rect.x + inset_x, rect.y + inset_top, max(1, rect.w - inset_x * 2), max(1, rect.h - inset_top - inset_bottom))]

    def _obstacles_ahead(self) -> list[SimObstacle]:
        player = self._player_rect()
        candidates = [obstacle for obstacle in self._obstacles if obstacle.right >= player[0] - 20]
        candidates.sort(key=lambda obstacle: max(0.0, obstacle.left - (player[0] + player[2])))
        return candidates

    def _get_observation(self) -> np.ndarray:
        player = self._player_rect()
        ahead = self._obstacles_ahead()
        nearest = ahead[0] if ahead else None
        second = ahead[1] if len(ahead) > 1 else None

        def obstacle_distance(obstacle: SimObstacle | None) -> float | None:
            if obstacle is None:
                return None
            return max(0.0, obstacle.left - player[0] - player[2])

        obs = build_dino_observation(
            player_y_px=self.player_y,
            player_vy_px=self.player_vy,
            grounded=self.grounded,
            ducking=self.ducking,
            current_speed_px_s=self.current_speed,
            screen_width_px=self.config.screen_width,
            screen_height_px=self.config.screen_height,
            floor_y_px=self.floor_y,
            max_speed_px_s=self.config.max_speed,
            nearest_distance_px=obstacle_distance(nearest),
            nearest_width_px=float(nearest.w) if nearest is not None else None,
            nearest_height_px=float(nearest.h) if nearest is not None else None,
            nearest_is_bird=bool(nearest.kind == "bird") if nearest is not None else False,
            second_distance_px=obstacle_distance(second),
            second_width_px=float(second.w) if second is not None else None,
            second_height_px=float(second.h) if second is not None else None,
            obstacle_count=len(ahead),
        )
        if self._episode_observation_noise > 0:
            noise = self.np_random.normal(0.0, self._episode_observation_noise, size=obs.shape).astype(np.float32)
            obs = np.clip(obs + noise, 0.0, 1.0).astype(np.float32)
        return self._delay_observation(obs)

    def _action_shaping_reward(self, action: int) -> float:
        if not self.config.enable_teacher_shaping:
            return 0.0

        teacher_action = DinoRulePolicy(DinoVisionConfig(min_action_interval_s=0.0)).decide(
            self._teacher_state(),
            now=float(self.elapsed_steps) * float(self.config.step_interval_s),
        ).action
        if teacher_action is PolicyAction.NOOP:
            if action == 1:
                return self.config.unnecessary_jump_penalty
            if action == 2:
                return self.config.unnecessary_duck_penalty
            return 0.0

        if action == int(self._policy_action_index(teacher_action)):
            return self.config.teacher_match_reward
        if action == 0:
            return self.config.missed_action_penalty
        return self.config.teacher_mismatch_penalty

    @staticmethod
    def _policy_action_index(action: PolicyAction) -> int:
        if action is PolicyAction.JUMP:
            return 1
        if action is PolicyAction.DUCK:
            return 2
        return 0

    def _teacher_state(self) -> VisionState:
        frame = np.zeros((self.config.screen_height, self.config.screen_width, 3), dtype=np.uint8)
        playfield = frame.copy()
        player_x, player_y, player_w, player_h = self._player_rect()
        player_box = Rect(int(player_x), int(player_y), int(player_w), int(player_h))
        obstacles = [self._teacher_obstacle(obstacle, player_box) for obstacle in self._obstacles]
        obstacles = [obstacle for obstacle in obstacles if obstacle is not None]
        nearest = obstacles[0] if obstacles else None
        return VisionState(
            timestamp=self._episode_elapsed_time,
            frame=frame,
            playfield=playfield,
            player_box=player_box,
            ground_y=int(self.floor_y),
            obstacles=obstacles,
            nearest_obstacle=nearest,
            nearest_distance_px=nearest.distance_px if nearest is not None else None,
            estimated_speed_px_s=self.current_speed,
            estimated_player_vy_px_s=float(self.player_vy),
            game_over=False,
        )

    @staticmethod
    def _teacher_obstacle(obstacle: SimObstacle, player_box: Rect) -> ObstacleDetection | None:
        rect = Rect(int(obstacle.left), int(obstacle.top), int(obstacle.w), int(obstacle.h))
        if rect.right < player_box.x - 40:
            return None
        return ObstacleDetection(
            rect=rect,
            label=obstacle.kind,
            distance_px=max(0.0, float(rect.x - player_box.right)),
            speed_px_s=0.0,
        )

    def _delay_observation(self, observation: np.ndarray) -> np.ndarray:
        if not self._observation_buffer:
            return observation

        self._observation_buffer.append(observation)
        return self._observation_buffer.pop(0)

    def _sample_episode_parameters(self) -> None:
        if not self.config.domain_randomization:
            self._episode_base_speed = float(self.config.base_speed)
            self._episode_speed_acceleration = float(self.config.speed_acceleration)
            self._episode_max_speed = float(self.config.max_speed)
            self._episode_gravity = float(self.config.gravity)
            self._episode_jump_velocity = float(self.config.jump_velocity)
            self._episode_bird_probability = float(self.config.bird_probability)
            self._episode_min_spawn_gap_px = int(self.config.min_spawn_gap_px)
            self._episode_max_spawn_gap_px = int(self.config.max_spawn_gap_px)
            self._episode_obstacle_scale = 1.0
            self._episode_observation_noise = float(self.config.observation_noise)
            return

        strength = float(max(0.0, self.config.domain_randomization_strength))
        self._episode_base_speed = self._sample_scaled_value(self.config.base_speed, strength)
        self._episode_speed_acceleration = self._sample_scaled_value(self.config.speed_acceleration, strength * 0.45)
        sampled_max_speed = self._sample_scaled_value(self.config.max_speed, strength * 0.2)
        self._episode_max_speed = max(self._episode_base_speed + 120.0, sampled_max_speed)
        self._episode_gravity = self._sample_scaled_value(self.config.gravity, strength * 0.35)
        self._episode_jump_velocity = self._sample_scaled_value(self.config.jump_velocity, strength * 0.28)
        bird_probability = self._sample_scaled_value(self.config.bird_probability, strength * 0.8)
        self._episode_bird_probability = float(np.clip(bird_probability, 0.02, 0.45))
        self._episode_min_spawn_gap_px = self._sample_scaled_gap(self.config.min_spawn_gap_px, strength * 1.1)
        self._episode_max_spawn_gap_px = self._sample_scaled_gap(self.config.max_spawn_gap_px, strength * 1.15)
        if self._episode_max_spawn_gap_px <= self._episode_min_spawn_gap_px:
            self._episode_max_spawn_gap_px = self._episode_min_spawn_gap_px + 20
        self._episode_obstacle_scale = self._sample_scaled_value(1.0, strength * 0.35)
        self._episode_observation_noise = float(self.config.observation_noise)

    def _sample_scaled_value(self, value: float, jitter: float) -> float:
        if jitter <= 0:
            return float(value)
        scale = float(self.np_random.uniform(1.0 - jitter, 1.0 + jitter))
        return float(value * scale)

    def _sample_scaled_gap(self, value: int, jitter: float) -> int:
        sample = int(round(self._sample_scaled_value(float(value), jitter)))
        return max(60, sample)

    def _sample_scaled_dimension(self, minimum: int, maximum: int, scale: float) -> int:
        low = max(6, int(round(minimum * scale)))
        high = max(low + 1, int(round(maximum * scale)))
        return int(self.np_random.integers(low, high + 1))

    def _render_frame(self) -> np.ndarray:
        width = self.config.screen_width
        height = self.config.screen_height
        frame = np.full((height, width, 3), 248, dtype=np.uint8)
        ground_y = self.floor_y
        cv2.line(frame, (0, ground_y), (width, ground_y), (40, 40, 40), 2)

        px, py, pw, ph = self._player_rect()
        cv2.rectangle(frame, (int(px), int(py)), (int(px + pw), int(py + ph)), (40, 40, 40), -1)
        if self.ducking and self.grounded:
            cv2.rectangle(frame, (int(px + 4), int(py + 4)), (int(px + pw - 4), int(py + ph - 6)), (80, 80, 80), -1)

        for obstacle in self._obstacles:
            color = (0, 120, 220) if obstacle.kind == "bird" else (30, 30, 30)
            cv2.rectangle(
                frame,
                (int(obstacle.left), int(obstacle.top)),
                (int(obstacle.right), int(obstacle.bottom)),
                color,
                -1,
            )

        cv2.putText(
            frame,
            f"speed={self.current_speed:.0f} score={self.score:.1f} passed={self.passed_obstacles}",
            (12, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        return frame


# Compatibility alias kept for older imports/tests while DinoEnv becomes the
# canonical Gymnasium environment used by the PPO path.
DinoSimEnv = DinoEnv
