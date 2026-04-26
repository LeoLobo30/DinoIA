from __future__ import annotations

import time
from dataclasses import dataclass
import threading

import cv2

from .capture import DinoScreenCapture
from .config import DinoVisionConfig, RealGameConfig
from .control import KeyboardController
from .decision import DinoRulePolicy
from .types import PolicyAction
from .vision import DinoVisionAnalyzer


@dataclass(slots=True)
class RunnerStats:
    frames: int = 0
    jumps: int = 0
    ducks: int = 0
    noops: int = 0


class DinoRealGameRunner:
    def __init__(
        self,
        real_config: RealGameConfig | None = None,
        vision_config: DinoVisionConfig | None = None,
        debug: bool | None = None,
    ) -> None:
        self.real_config = real_config or RealGameConfig()
        self.vision_config = vision_config or DinoVisionConfig()
        if debug is not None:
            self.real_config.debug = debug

        self.capture = DinoScreenCapture(self.real_config)
        self.analyzer = DinoVisionAnalyzer(self.vision_config)
        self.policy = DinoRulePolicy(self.vision_config)
        self.controller = KeyboardController()
        self.stats = RunnerStats()

    def run(self, duration_s: float | None = None) -> RunnerStats:
        stop_event = threading.Event()
        return self.run_until(duration_s=duration_s, stop_event=stop_event)

    def run_until(self, duration_s: float | None = None, stop_event: threading.Event | None = None) -> RunnerStats:
        last_focus_at = 0.0
        if self.real_config.focus_window:
            try:
                self.capture.focus_window()
                last_focus_at = time.time()
            except Exception:
                pass

        if self.real_config.auto_start:
            time.sleep(0.5)
            self.controller.jump()

        start = time.time()
        frame_interval = 1.0 / max(1.0, self.real_config.frame_rate)

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break

                now = time.time()
                if duration_s is not None and now - start >= duration_s:
                    break

                if (
                    self.real_config.focus_window
                    and self.real_config.keep_window_foreground
                    and now - last_focus_at >= self.real_config.focus_refresh_s
                ):
                    try:
                        self.capture.focus_window()
                        last_focus_at = now
                    except Exception:
                        pass

                frame = self.capture.capture()
                state = self.analyzer.analyze(frame, timestamp=now)

                if self.real_config.auto_restart and state.game_over:
                    self.controller.release_duck()
                    self.controller.restart()
                    time.sleep(0.6)
                    continue

                decision = self.policy.decide(state, now=now)
                control_result = self.controller.perform(decision.action)
                self._count_action(control_result.action)

                if self.real_config.debug:
                    overlay = self.analyzer.draw_overlay(state)
                    cv2.imshow("DinoIA - Real Game", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                self.stats.frames += 1
                sleep_for = max(0.0, frame_interval - (time.time() - now))
                time.sleep(sleep_for)
        finally:
            self.controller.release_duck()
            if self.real_config.debug:
                cv2.destroyAllWindows()

        return self.stats

    def _count_action(self, action: PolicyAction) -> None:
        if action is PolicyAction.JUMP:
            self.stats.jumps += 1
        elif action is PolicyAction.DUCK:
            self.stats.ducks += 1
        else:
            self.stats.noops += 1


