from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import DinoVisionConfig
from .types import ObstacleDetection, Rect, VisionState


@dataclass(slots=True)
class _TrackerState:
    last_obstacle_x: float | None = None
    last_obstacle_label: str | None = None
    last_obstacle_rect: Rect | None = None
    last_timestamp: float | None = None
    smoothed_speed: float = 0.0
    player_box: Rect | None = None
    last_player_y: float | None = None
    last_player_timestamp: float | None = None
    smoothed_player_vy: float = 0.0


class DinoVisionAnalyzer:
    def __init__(self, config: DinoVisionConfig | None = None):
        self.config = config or DinoVisionConfig()
        self._tracker = _TrackerState()

    def analyze(self, frame: np.ndarray, timestamp: float | None = None) -> VisionState:
        if timestamp is None:
            timestamp = time.time()

        playfield = self._crop_playfield(frame)
        gray = self._ensure_gray(playfield)
        binary = self._build_binary_mask(gray)
        ground_y = self._estimate_ground_line(binary)
        player_box = self._detect_player_box(binary, ground_y)
        player_velocity = self._estimate_player_velocity(player_box, timestamp)
        obstacles = self._detect_obstacles(binary, ground_y, player_box)
        nearest = self._pick_nearest_obstacle(obstacles)
        nearest_distance = nearest.distance_px if nearest is not None else None
        estimated_speed = self._estimate_speed(nearest, timestamp)
        game_over = self._detect_game_over(binary)

        return VisionState(
            timestamp=timestamp,
            frame=frame,
            playfield=playfield,
            player_box=player_box,
            ground_y=ground_y,
            obstacles=obstacles,
            nearest_obstacle=nearest,
            nearest_distance_px=nearest_distance,
            estimated_speed_px_s=estimated_speed,
            estimated_player_vy_px_s=player_velocity,
            game_over=game_over,
        )

    def draw_overlay(self, state: VisionState) -> np.ndarray:
        frame = state.playfield.copy()
        cv2.line(frame, (0, state.ground_y), (frame.shape[1], state.ground_y), (0, 0, 255), 1)

        self._draw_rect(frame, state.player_box, (0, 180, 0), "player")
        for obstacle in state.obstacles:
            color = (255, 0, 0) if obstacle is not state.nearest_obstacle else (0, 165, 255)
            self._draw_rect(frame, obstacle.rect, color, f"{obstacle.label}:{int(obstacle.distance_px)}px")

        text_lines = [
            f"speed: {state.estimated_speed_px_s:.1f}px/s",
            f"distance: {state.nearest_distance_px:.1f}px" if state.nearest_distance_px is not None else "distance: none",
        ]
        for index, text in enumerate(text_lines):
            cv2.putText(
                frame,
                text,
                (10, 22 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                text,
                (10, 22 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )

        return frame

    def _crop_playfield(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        top = int(height * self.config.playfield_top_ratio)
        bottom = int(height * self.config.playfield_bottom_ratio)
        left = int(width * self.config.playfield_left_ratio)
        right = int(width * self.config.playfield_right_ratio)
        return frame[top:bottom, left:right]

    @staticmethod
    def _ensure_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _build_binary_mask(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        mean_brightness = float(np.mean(blurred))
        if mean_brightness >= 127.0:
            _, binary = cv2.threshold(blurred, self.config.dark_threshold, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binary = cv2.threshold(blurred, self.config.light_threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def _estimate_ground_line(self, binary: np.ndarray) -> int:
        height = binary.shape[0]
        start_row = int(height * self.config.ground_search_fraction)
        search = binary[start_row:, :]
        row_sums = search.sum(axis=1).astype(np.float32)
        if row_sums.size == 0:
            return int(height * 0.8)

        smoothed = np.convolve(row_sums, np.ones(7, dtype=np.float32) / 7.0, mode="same")
        index = int(np.argmax(smoothed))
        return start_row + index

    def _detect_player_box(self, binary: np.ndarray, ground_y: int) -> Rect:
        height, width = binary.shape[:2]
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[Rect] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_player_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            rect = Rect(x, y, w, h)
            if rect.x > width * self.config.player_search_fraction:
                continue
            if rect.x > min(self.config.max_player_x_px, int(width * self.config.max_player_x_fraction)):
                continue
            if rect.bottom < ground_y - self.config.player_ground_band_px:
                continue
            if rect.bottom > ground_y + 18:
                continue
            if rect.w > min(self.config.max_player_width_px, int(width * 0.18)):
                continue
            if rect.h > min(self.config.max_player_height_px, int(height * 0.45)):
                continue
            if rect.w < self.config.min_player_width_px or rect.h < self.config.min_player_height_px:
                continue
            if rect.w / max(1, rect.h) > self.config.max_player_aspect_ratio:
                continue
            if (
                rect.h >= self.config.min_standing_player_height_px
                and rect.w > self.config.max_standing_player_width_px
            ):
                continue
            if (
                self.config.max_ducking_player_height_px
                < rect.h
                < self.config.min_standing_player_height_px
            ):
                continue
            candidates.append(rect)

        if candidates:
            if self._tracker.player_box is not None:
                previous = self._tracker.player_box
                max_center_jump = float(self.config.max_player_center_jump_px)
                stable_candidates = [
                    rect
                    for rect in candidates
                    if abs(rect.center_x - previous.center_x) <= max_center_jump
                    and abs(rect.center_y - previous.center_y) <= self.config.max_player_vertical_jump_px
                ]
                if stable_candidates:
                    candidates = stable_candidates
                else:
                    return previous
            best = min(candidates, key=lambda rect: (ground_y - rect.bottom if rect.bottom <= ground_y else 999, rect.x, -rect.area))
            self._tracker.player_box = best
            return best

        if self._tracker.player_box is not None:
            return self._tracker.player_box

        fallback_height = max(32, int(height * 0.22))
        fallback_width = max(24, int(width * 0.05))
        fallback_x = int(width * 0.12)
        fallback_y = max(0, ground_y - fallback_height)
        fallback = Rect(fallback_x, fallback_y, fallback_width, fallback_height)
        self._tracker.player_box = fallback
        return fallback

    def _detect_obstacles(self, binary: np.ndarray, ground_y: int, player_box: Rect) -> list[ObstacleDetection]:
        height, width = binary.shape[:2]
        obstacle_mask = self._remove_ground_line(binary, ground_y)
        contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obstacles: list[ObstacleDetection] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_obstacle_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            rect = Rect(x, y, w, h)
            if self._looks_like_player(rect, player_box):
                continue
            if rect.x < player_box.right + self.config.obstacle_player_clearance_px:
                continue
            if self._looks_like_ground_artifact(rect, ground_y, width, height):
                continue
            if rect.bottom < ground_y - self.config.obstacle_top_margin_px:
                continue
            if rect.y < int(height * 0.18) and rect.bottom < ground_y - 24:
                continue
            if rect.w < 8 or rect.h < 8:
                continue

            label = "bird" if rect.bottom < ground_y - self.config.bird_height_threshold_px else "cactus"
            if label == "bird" and (rect.w < self.config.min_bird_width_px or rect.h < self.config.min_bird_height_px):
                continue
            distance = max(0.0, float(rect.x - player_box.right))
            obstacles.append(ObstacleDetection(rect=rect, label=label, distance_px=distance))

        obstacles.sort(key=lambda obstacle: obstacle.distance_px)
        return obstacles

    def _remove_ground_line(self, binary: np.ndarray, ground_y: int) -> np.ndarray:
        cleaned = binary.copy()
        margin = max(1, self.config.ground_line_cleanup_px)
        top = max(0, ground_y - margin)
        bottom = min(cleaned.shape[0], ground_y + margin + 1)
        cleaned[top:bottom, :] = 0
        return cleaned

    def _looks_like_ground_artifact(self, rect: Rect, ground_y: int, width: int, height: int) -> bool:
        if rect.w > int(width * self.config.max_obstacle_width_fraction):
            return True
        if rect.h <= 16 and rect.bottom >= ground_y - self.config.ground_line_cleanup_px * 3:
            return True
        if rect.w > rect.h * 4 and rect.bottom >= ground_y - int(height * 0.08):
            return True
        return False

    @staticmethod
    def _intersection_area(first: Rect, second: Rect) -> int:
        left = max(first.x, second.x)
        top = max(first.y, second.y)
        right = min(first.right, second.right)
        bottom = min(first.bottom, second.bottom)
        return max(0, right - left) * max(0, bottom - top)

    def _looks_like_player(self, rect: Rect, player_box: Rect) -> bool:
        overlap = self._intersection_area(rect, player_box)
        if overlap <= 0:
            return False

        overlap_ratio = overlap / max(1, min(rect.area, player_box.area))
        same_left_cluster = rect.x <= player_box.right + self.config.obstacle_side_margin_px
        contained_by_player = rect.x >= player_box.x - 4 and rect.right <= player_box.right + 8
        return overlap_ratio >= 0.35 and (same_left_cluster or contained_by_player)

    @staticmethod
    def _pick_nearest_obstacle(obstacles: list[ObstacleDetection]) -> ObstacleDetection | None:
        if not obstacles:
            return None
        return obstacles[0]

    def _estimate_speed(self, nearest: ObstacleDetection | None, timestamp: float) -> float:
        if nearest is None:
            self._tracker.last_timestamp = timestamp
            self._tracker.last_obstacle_x = None
            self._tracker.last_obstacle_label = None
            self._tracker.last_obstacle_rect = None
            self._tracker.smoothed_speed *= 0.85
            return self._tracker.smoothed_speed

        previous_rect = self._tracker.last_obstacle_rect
        previous_label = self._tracker.last_obstacle_label
        if previous_rect is not None and previous_label == nearest.label and self._tracker.last_timestamp is not None:
            same_obstacle = (
                abs(previous_rect.y - nearest.rect.y) <= 18
                and abs(previous_rect.w - nearest.rect.w) <= 24
                and abs(previous_rect.h - nearest.rect.h) <= 24
            )
            if same_obstacle:
                dt = max(1e-3, timestamp - self._tracker.last_timestamp)
                measured_speed = max(0.0, (self._tracker.last_obstacle_x - nearest.rect.x) / dt)
                if self._tracker.smoothed_speed <= 0:
                    self._tracker.smoothed_speed = measured_speed
                else:
                    alpha = self.config.speed_smoothing
                    self._tracker.smoothed_speed = alpha * measured_speed + (1.0 - alpha) * self._tracker.smoothed_speed
            else:
                self._tracker.smoothed_speed *= 0.9

        self._tracker.last_obstacle_x = float(nearest.rect.x)
        self._tracker.last_obstacle_label = nearest.label
        self._tracker.last_obstacle_rect = nearest.rect
        self._tracker.last_timestamp = timestamp
        return self._tracker.smoothed_speed

    def _estimate_player_velocity(self, player_box: Rect, timestamp: float) -> float:
        previous_y = self._tracker.last_player_y
        previous_timestamp = self._tracker.last_player_timestamp
        vy = 0.0

        if previous_y is not None and previous_timestamp is not None:
            dt = max(1e-3, timestamp - previous_timestamp)
            vy = (player_box.y - previous_y) / dt
            alpha = self.config.speed_smoothing
            self._tracker.smoothed_player_vy = alpha * vy + (1.0 - alpha) * self._tracker.smoothed_player_vy
        else:
            self._tracker.smoothed_player_vy *= 0.9

        self._tracker.last_player_y = float(player_box.y)
        self._tracker.last_player_timestamp = timestamp
        return self._tracker.smoothed_player_vy

    @staticmethod
    def _draw_rect(frame: np.ndarray, rect: Rect, color: tuple[int, int, int], label: str) -> None:
        cv2.rectangle(frame, (rect.x, rect.y), (rect.right, rect.bottom), color, 2)
        cv2.putText(
            frame,
            label,
            (rect.x, max(16, rect.y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _detect_game_over(binary: np.ndarray) -> bool:
        height, width = binary.shape[:2]
        top = binary[: int(height * 0.35), int(width * 0.22) : int(width * 0.78)]
        if top.size == 0:
            return False

        dark_ratio = float(np.count_nonzero(top)) / float(top.size)
        return dark_ratio > 0.045
