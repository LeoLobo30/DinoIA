from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


@dataclass(frozen=True, slots=True)
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

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.h / 2.0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def expand(self, margin: int) -> "Rect":
        return Rect(self.x - margin, self.y - margin, self.w + margin * 2, self.h + margin * 2)

    def clipped(self, width: int, height: int) -> "Rect":
        x = max(0, self.x)
        y = max(0, self.y)
        right = min(width, self.right)
        bottom = min(height, self.bottom)
        return Rect(x, y, max(0, right - x), max(0, bottom - y))


class PolicyAction(str, Enum):
    NOOP = "noop"
    JUMP = "jump"
    DUCK = "duck"


@dataclass(frozen=True, slots=True)
class ObstacleDetection:
    rect: Rect
    label: str
    distance_px: float
    confidence: float = 1.0
    speed_px_s: float = 0.0


@dataclass(slots=True)
class VisionState:
    timestamp: float
    frame: np.ndarray
    playfield: np.ndarray
    player_box: Rect
    ground_y: int
    obstacles: list[ObstacleDetection]
    nearest_obstacle: Optional[ObstacleDetection]
    nearest_distance_px: Optional[float]
    estimated_speed_px_s: float
    estimated_player_vy_px_s: float = 0.0
    game_over: bool = False


@dataclass(frozen=True, slots=True)
class ScreenRegion:
    left: int
    top: int
    width: int
    height: int

    def as_mss(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }
