from __future__ import annotations

import time
from dataclasses import dataclass

from .config import DinoVisionConfig
from .types import PolicyAction, VisionState


@dataclass(slots=True)
class PolicyDecision:
    action: PolicyAction
    reason: str


class DinoRulePolicy:
    def __init__(self, config: DinoVisionConfig | None = None):
        self.config = config or DinoVisionConfig()
        self._last_jump_time = 0.0

    def decide(self, state: VisionState, now: float | None = None) -> PolicyDecision:
        if now is None:
            now = time.time()

        obstacle = state.nearest_obstacle
        if obstacle is None or state.nearest_distance_px is None:
            return PolicyDecision(action=PolicyAction.NOOP, reason="no_obstacle")

        if obstacle.label == "bird" and state.nearest_distance_px <= self.config.duck_distance_px:
            return PolicyDecision(action=PolicyAction.DUCK, reason="bird_close")

        dynamic_threshold = self.config.jump_base_distance_px + min(
            state.estimated_speed_px_s * self.config.jump_speed_factor,
            self.config.jump_base_distance_px * 2.2,
        )

        if state.nearest_distance_px <= dynamic_threshold and now - self._last_jump_time >= self.config.min_action_interval_s:
            self._last_jump_time = now
            return PolicyDecision(action=PolicyAction.JUMP, reason=f"distance={state.nearest_distance_px:.1f}")

        return PolicyDecision(action=PolicyAction.NOOP, reason="safe")
