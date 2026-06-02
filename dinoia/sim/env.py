from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import SimConfig
from ..observations import OBSERVATION_SIZE, build_dino_observation


@dataclass(slots=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


@dataclass(slots=True)
class SimObstacle:
    kind: str
    variant: str
    x: float
    y: float
    w: int
    h: int
    speed_offset_units: float = 0.0
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
        self._episode_index = -1
        self._last_spawn_kind: str | None = None
        self._same_kind_streak = 0
        self._last_spawn_variant: str | None = None
        self._recent_spawn_variants: list[str] = []
        self._distance_run = 0.0
        self._anim_time = 0.0
        self._player_anim_frame = 0
        self._bird_anim_time = 0.0
        self._bird_anim_frame = 0
        self._action_buffer: list[int] = []
        self._observation_buffer: list[np.ndarray] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._episode_index += 1
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
        self._last_spawn_kind = None
        self._same_kind_streak = 0
        self._last_spawn_variant = None
        self._recent_spawn_variants = []
        self._distance_run = 0.0
        self._anim_time = 0.0
        self._player_anim_frame = 0
        self._bird_anim_time = 0.0
        self._bird_anim_frame = 0
        self._sample_episode_parameters()
        self._spawn_timer = max(float(self.config.initial_obstacle_delay_s), self._next_spawn_delay())
        observation = self._get_observation()
        self._reset_latency_buffers(observation)
        info = {"speed": self.current_speed, "passed_obstacles": self.passed_obstacles}
        return observation, info

    def step(self, action: int):
        self.elapsed_steps += 1
        policy_dt = self._sample_policy_dt()
        physics_dt = float(self.config.physics_step_interval_s)
        substeps = max(1, int(round(policy_dt / max(1e-6, physics_dt))))
        requested_action = int(action)
        executed_action = self._delay_action(requested_action)
        self._apply_action(executed_action)

        collision = False
        passed_now = False
        for _ in range(substeps):
            self._episode_elapsed_time += physics_dt
            self._update_animation_state(physics_dt)
            self._advance_player(physics_dt)
            self._advance_obstacles(physics_dt)
            self._distance_run += self.current_speed * physics_dt * float(self.config.distance_scale_per_step)
            self._maybe_spawn_obstacle(physics_dt)
            passed_now = self._consume_pass_rewards() or passed_now
            collision = self._check_collision()
            if collision:
                break

        reward = self.config.survival_reward
        if passed_now:
            reward += self._pass_obstacle_reward()
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
        # Closer to Chromium progression: smooth acceleration + score-linked discrete increments.
        smooth = self._episode_base_speed + self._episode_elapsed_time * self._episode_speed_acceleration
        score_steps = int(self._distance_run / max(1.0, float(self.config.speed_increment_interval_score)))
        step_boost = score_steps * float(self.config.speed_increment) * float(self.config.speed_unit_px_s) * 0.05
        speed = smooth + step_boost
        return float(min(self._episode_max_speed, speed))

    def spawn_obstacle(
        self,
        kind: str = "cactus",
        variant: str | None = None,
        *,
        x: float | None = None,
        width: int | None = None,
        height: int | None = None,
        y: float | None = None,
        speed_offset_units: float = 0.0,
    ) -> SimObstacle:
        x = float(self.config.screen_width + 15 if x is None else x)
        screen_scale = float(self.config.screen_height) / 150.0
        if kind == "bird":
            variant = variant or "bird"
            # Chromium sprite definition (normal mode): bird = 46x40, y in {100, 75, 50}.
            base_width = max(6, int(round(46 * screen_scale)))
            base_height = max(6, int(round(40 * screen_scale)))
            width = width or max(6, int(round(base_width * self._episode_obstacle_scale)))
            height = height or max(6, int(round(base_height * self._episode_obstacle_scale)))
            if y is None:
                speed_units = self.current_speed / max(1.0, float(self.config.speed_unit_px_s))
                if speed_units < 10.0:
                    base_levels = [100, 75]
                else:
                    base_levels = [100, 75, 50]
                level = int(self.np_random.choice(base_levels))
                level = int(round(level * screen_scale))
                y = float(max(0, min(self.floor_y - 8, level)))
            else:
                y = float(y)
        else:
            # Chromium sprite definitions (normal mode):
            # CACTUS_SMALL = 17x35, CACTUS_LARGE = 25x50.
            cactus_specs = [("cactus_small", 17, 35), ("cactus_large", 25, 50)]
            if width is None or height is None:
                picked_variant, base_w, base_h = cactus_specs[int(self.np_random.integers(0, len(cactus_specs)))]
                variant = variant or picked_variant
                base_w = max(6, int(round(base_w * screen_scale)))
                base_h = max(6, int(round(base_h * screen_scale)))
                width = width or max(6, int(round(base_w * self._episode_obstacle_scale)))
                height = height or max(6, int(round(base_h * self._episode_obstacle_scale)))
            else:
                variant = variant or "cactus_custom"
            y = float(y if y is not None else self.floor_y - height)

        obstacle = SimObstacle(
            kind=kind,
            variant=variant or kind,
            x=x,
            y=y,
            w=width,
            h=height,
            speed_offset_units=float(speed_offset_units),
        )
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
        for obstacle in self._obstacles:
            # Chromium-like per-obstacle speed offset (notably birds).
            obstacle_speed = self.current_speed + obstacle.speed_offset_units * float(self.config.speed_unit_px_s)
            obstacle.x -= max(1.0, obstacle_speed) * dt

    def _maybe_spawn_obstacle(self, dt: float) -> None:
        self._spawn_timer -= dt
        if self._spawn_timer > 0:
            return
        if len(self._obstacles) >= int(self.config.max_obstacles_on_screen):
            self._spawn_timer = 0.08
            return
        kind = self._select_spawn_kind()
        obstacle = self._spawn_with_chromium_rules(kind)
        self._record_spawn_kind(kind)
        self._spawn_timer = self._next_spawn_delay(obstacle)

    def _select_spawn_kind(self) -> str:
        # Chromium-like bird gating: appears only at higher speeds.
        bird_min_speed = float(self.config.bird_min_speed_units) * float(self.config.speed_unit_px_s)
        can_spawn_bird = self.current_speed >= bird_min_speed
        choose_bird = can_spawn_bird and self.np_random.random() < self._episode_bird_probability
        if choose_bird and self.config.prohibit_bird_after_bird and self._last_spawn_kind == "bird":
            choose_bird = False
        candidate = "bird" if choose_bird else "cactus"
        # Avoid long same-kind runs (similar intent to Chromium duplicate checks).
        if self._same_kind_streak >= int(self.config.max_same_obstacle_streak) and self._last_spawn_kind == candidate:
            if candidate == "bird":
                return "cactus"
            if can_spawn_bird:
                return "bird"
        if candidate == "cactus" and self._last_spawn_variant == "cactus_large" and self.np_random.random() < 0.35:
            return "bird" if can_spawn_bird else "cactus"
        if candidate == "cactus" and self._is_recent_duplicate("cactus_large") and self.np_random.random() < 0.5:
            return "bird" if can_spawn_bird else "cactus"
        if candidate == "bird" and self._same_kind_streak >= int(self.config.max_bird_streak):
            return "cactus"
        return candidate

    def _spawn_with_chromium_rules(self, kind: str) -> SimObstacle:
        if kind == "bird":
            return self.spawn_obstacle(kind="bird", variant="bird", speed_offset_units=0.8)

        # Chromium-like cactus grouping at higher speeds (multipleSpeed around 4.5-7).
        speed_units = self.current_speed / max(1.0, float(self.config.speed_unit_px_s))
        if self._last_spawn_kind == "cactus" and self.np_random.random() < 0.6:
            choose_small = False
        else:
            choose_small = bool(self.np_random.random() < 0.5)
        if self._is_recent_duplicate("cactus_small") and choose_small:
            choose_small = False
        if self._is_recent_duplicate("cactus_large") and not choose_small:
            choose_small = True
        base_variant, base_w, base_h = ("cactus_small", 17, 35) if choose_small else ("cactus_large", 25, 50)
        screen_scale = float(self.config.screen_height) / 150.0
        base_w = max(6, int(round(base_w * screen_scale * self._episode_obstacle_scale)))
        base_h = max(6, int(round(base_h * screen_scale * self._episode_obstacle_scale)))

        group_len = 1
        if speed_units >= float(self.config.cactus_cluster_high_speed_units):
            group_len = int(self.np_random.integers(1, int(self.config.cactus_cluster_max_len_high_speed) + 1))
        elif speed_units >= float(self.config.cactus_cluster_min_speed_units):
            group_len = int(self.np_random.integers(1, int(self.config.cactus_cluster_max_len_mid_speed) + 1))

        # Chromium-like cactus grouping: spawn individual obstacles in a cluster.
        gap_between = max(2, int(round(3 * screen_scale)))
        first = self.spawn_obstacle(kind="cactus", variant=base_variant, width=base_w, height=base_h)
        last_right = first.right
        for _ in range(group_len - 1):
            x = float(last_right + gap_between)
            follower = self.spawn_obstacle(kind="cactus", variant=base_variant, x=x, width=base_w, height=base_h)
            last_right = follower.right

        self._last_spawn_variant = base_variant
        self._remember_spawn_variant(base_variant)
        # Use total cluster span as reference for subsequent gap timing.
        cluster_span = int(last_right - first.left)
        return SimObstacle(kind="cactus", variant=base_variant, x=first.x, y=first.y, w=cluster_span, h=first.h)

    def _record_spawn_kind(self, kind: str) -> None:
        if self._last_spawn_kind == kind:
            self._same_kind_streak += 1
        else:
            self._same_kind_streak = 1
        self._last_spawn_kind = kind
        if kind == "bird":
            self._last_spawn_variant = "bird"
            self._remember_spawn_variant("bird")

    def _sample_spawn_gap(self) -> float:
        min_gap = self._episode_min_spawn_gap_px
        max_gap = self._episode_max_spawn_gap_px
        if max_gap <= min_gap:
            max_gap = min_gap + 20
        return float(self.np_random.integers(min_gap, max_gap + 1))

    def _next_spawn_delay(self, obstacle: SimObstacle | None = None) -> float:
        if obstacle is None:
            gap_px = self._sample_spawn_gap() * float(self.config.gap_coefficient)
            return gap_px / max(1.0, self.current_speed)

        # Chromium-like gap model:
        # min_gap ~= obstacle_width * speed + type_min_gap * gap_coefficient,
        # with speed measured in Dino "speed units" (px/frame scale).
        speed_units = self.current_speed / max(1.0, float(self.config.speed_unit_px_s))
        speed_level = min(1.0, speed_units / 12.0)
        type_min_gap = 150.0 if obstacle.kind == "bird" else 120.0
        type_min_gap *= 1.0 - 0.18 * speed_level
        min_gap_px = obstacle.w * speed_units + type_min_gap * float(self.config.gap_coefficient)
        max_gap_px = min_gap_px * (1.55 - 0.15 * speed_level)
        gap_px = float(self.np_random.uniform(min_gap_px, max_gap_px))
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
        width = self.config.duck_width if self.ducking and self.grounded else self.config.player_width
        return float(self.config.player_x), float(self.player_y), float(width), float(height)

    @property
    def player_height(self) -> int:
        return self.config.duck_height if self.ducking and self.grounded else self.config.stand_height

    def _player_collision_boxes(self) -> list[Rect]:
        px, py, _, _ = self._player_rect()
        if self.ducking and self.grounded:
            return [Rect(int(px + 1), int(py + 18), 55, 25)]
        # Chromium running collision boxes (trex.ts).
        return [
            Rect(int(px + 22), int(py + 0), 17, 16),
            Rect(int(px + 1), int(py + 18), 30, 9),
            Rect(int(px + 10), int(py + 35), 14, 8),
            Rect(int(px + 1), int(py + 24), 29, 5),
            Rect(int(px + 5), int(py + 30), 21, 4),
            Rect(int(px + 9), int(py + 34), 15, 4),
        ]

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
            return [
                Rect(rect.x + 15, rect.y + 15, 16, 5),
                Rect(rect.x + 18, rect.y + 21, 24, 6),
                Rect(rect.x + 2, rect.y + 14, 4, 3),
                Rect(rect.x + 6, rect.y + 10, 4, 7),
                Rect(rect.x + 10, rect.y + 8, 6, 9),
            ]

        if obstacle.variant == "cactus_large":
            return [
                Rect(rect.x + 0, rect.y + 12, 7, 38),
                Rect(rect.x + 8, rect.y + 0, max(1, rect.w - 8 - 10), 49),
                Rect(rect.x + max(0, rect.w - 10), rect.y + 10, 10, 38),
            ]
        if obstacle.variant == "cactus_small":
            return [
                Rect(rect.x + 0, rect.y + 7, 5, 27),
                Rect(rect.x + 4, rect.y + 0, max(1, rect.w - 4 - 7), 34),
                Rect(rect.x + max(0, rect.w - 7), rect.y + 4, 7, 14),
            ]

        inset_x = max(1, rect.w // 8)
        inset_top = max(1, rect.h // 12)
        inset_bottom = max(1, rect.h // 10)
        return [Rect(rect.x + inset_x, rect.y + inset_top, max(1, rect.w - inset_x * 2), max(1, rect.h - inset_top - inset_bottom))]

    def _update_animation_state(self, dt: float) -> None:
        self._anim_time += dt * 1000.0
        player_frame_ms = (1000.0 / 8.0) if self.ducking and self.grounded else (1000.0 / 12.0)
        while self._anim_time >= player_frame_ms:
            self._anim_time -= player_frame_ms
            self._player_anim_frame = (self._player_anim_frame + 1) % 2

        self._bird_anim_time += dt * 1000.0
        while self._bird_anim_time >= (1000.0 / 6.0):
            self._bird_anim_time -= (1000.0 / 6.0)
            self._bird_anim_frame = (self._bird_anim_frame + 1) % 2

    def _remember_spawn_variant(self, variant: str) -> None:
        self._recent_spawn_variants.append(variant)
        max_len = max(1, int(self.config.recent_obstacle_memory))
        if len(self._recent_spawn_variants) > max_len:
            self._recent_spawn_variants = self._recent_spawn_variants[-max_len:]

    def _is_recent_duplicate(self, variant: str) -> bool:
        return self._recent_spawn_variants.count(variant) >= 2

    def _sample_policy_dt(self) -> float:
        base_dt = float(self.config.step_interval_s)
        jitter = max(0.0, float(self.config.frame_time_jitter))
        dt = base_dt
        if jitter > 0:
            dt *= float(self.np_random.uniform(1.0 - jitter, 1.0 + jitter))
        if self.np_random.random() < max(0.0, float(self.config.frame_time_spike_probability)):
            dt *= max(1.0, float(self.config.frame_time_spike_scale))
        return float(np.clip(dt, base_dt * 0.5, base_dt * 2.5))

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
        if action == 1 and not self.grounded:
            return self.config.unnecessary_jump_penalty
        if action == 2 and not self.grounded:
            return self.config.unnecessary_duck_penalty
        if action != 0:
            return self.config.action_penalty
        return 0

    def _pass_obstacle_reward(self) -> float:
        speed_units = self.current_speed / max(1.0, float(self.config.speed_unit_px_s))
        return float(self.config.pass_reward + self.config.speed_pass_reward_scale * speed_units)

    def _delay_observation(self, observation: np.ndarray) -> np.ndarray:
        if not self._observation_buffer:
            return observation

        self._observation_buffer.append(observation)
        return self._observation_buffer.pop(0)

    def _sample_episode_parameters(self) -> None:
        if self.config.curriculum_learning:
            speed_min, speed_max = self._curriculum_speed_range()
            self._episode_base_speed = float(self.np_random.uniform(speed_min, speed_max))
            self._episode_speed_acceleration = float(self.config.speed_acceleration)
            self._episode_max_speed = float(max(self._episode_base_speed, speed_max))
            self._episode_gravity = float(self.config.gravity)
            self._episode_jump_velocity = float(self.config.jump_velocity)
            self._episode_bird_probability = float(self.config.bird_probability)
            self._episode_min_spawn_gap_px = int(self.config.min_spawn_gap_px)
            self._episode_max_spawn_gap_px = int(self.config.max_spawn_gap_px)
            self._episode_obstacle_scale = 1.0
            self._episode_observation_noise = float(self.config.observation_noise)
            return

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

    def _curriculum_speed_range(self) -> tuple[float, float]:
        episode = max(0, self._episode_index)
        if episode < self.config.curriculum_low_episodes:
            return self.config.curriculum_low_speed_min, self.config.curriculum_low_speed_max
        if episode < self.config.curriculum_medium_episodes:
            return self.config.curriculum_medium_speed_min, self.config.curriculum_medium_speed_max
        if episode < self.config.curriculum_high_episodes:
            return self.config.curriculum_high_speed_min, self.config.curriculum_high_speed_max
        return self.config.curriculum_random_speed_min, self.config.curriculum_random_speed_max

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


DinoSimEnv = DinoEnv
