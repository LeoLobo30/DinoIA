from __future__ import annotations

from dataclasses import dataclass

from .config import SimConfig
from .observations import build_dino_observation
from .types import PolicyAction, Rect, VisionState


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
        nearest_distance = self._scale_x(nearest.distance_px, x_scale) if nearest is not None else None
        second_distance = self._scale_x(second.distance_px, x_scale) if second is not None else None
        current_speed = self._estimate_current_speed(state, x_scale)
        player_vy = self._scale_y(state.estimated_player_vy_px_s, y_scale)
        duck_height = self._scale_y(player.h, y_scale)
        grounded = player.bottom >= state.ground_y - 4

        return build_dino_observation(
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

    def estimate_current_speed_px_s(self, state: VisionState) -> float:
        x_scale = self._x_scale(state.playfield.shape[1])
        return self._estimate_current_speed(state, x_scale)

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
