from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import RealGameConfig
from .types import ScreenRegion


class DinoScreenCapture:
    def __init__(self, config: RealGameConfig):
        self.config = config
        self._region_cache: ScreenRegion | None = None
        self._mss = None

    def resolve_region(self) -> ScreenRegion:
        if self.config.manual_region is not None:
            left, top, width, height = self.config.manual_region
            return ScreenRegion(left=left, top=top, width=width, height=height)

        if self._region_cache is not None:
            return self._region_cache

        region = self._find_window_region(self.config.window_title_candidates)
        if region is None:
            raise RuntimeError(
                "Could not resolve the Chrome Dino window. "
                "Open the game in Chrome, maximize the window, or pass a manual region."
            )

        self._region_cache = region
        return region

    def focus_window(self) -> None:
        try:
            import pygetwindow as gw
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("pygetwindow is required to focus the Dino window.") from exc

        window = self._find_window(self.config.window_title_candidates, gw)
        if window is None:
            raise RuntimeError("Chrome Dino window not found.")

        try:
            window.activate()
        except Exception:
            try:
                window.restore()
            except Exception:
                pass
            try:
                window.minimize()
                window.restore()
            except Exception:
                pass
            try:
                window.activate()
            except Exception:
                self._force_foreground_window(window)

    @staticmethod
    def _force_foreground_window(window: object) -> None:
        try:
            import ctypes
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("ctypes is required to force window focus.") from exc

        hwnd = getattr(window, "_hWnd", None) or getattr(window, "_hWnd", None)
        if hwnd is None:
            hwnd = getattr(window, "hWnd", None)
        if hwnd is None:
            return

        user32 = ctypes.windll.user32
        SW_RESTORE = 9
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.SetForegroundWindow(hwnd)
        user32.SetActiveWindow(hwnd)

    def capture(self) -> np.ndarray:
        region = self.resolve_region()
        sct = self._ensure_mss()
        shot = sct.grab(region.as_mss())
        frame = np.array(shot, dtype=np.uint8)
        return frame[:, :, :3].copy()

    def close(self) -> None:
        mss_instance = self._mss
        self._mss = None
        if mss_instance is None:
            return

        close = getattr(mss_instance, "close", None)
        if callable(close):
            close()

    def _ensure_mss(self):
        if self._mss is not None:
            return self._mss

        try:
            from mss import mss
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("mss is required for screen capture.") from exc

        self._mss = mss()
        return self._mss

    def _find_window_region(self, candidates: Sequence[str]) -> ScreenRegion | None:
        try:
            import pygetwindow as gw
        except Exception:  # pragma: no cover - optional runtime dependency
            return None

        window = self._find_window(candidates, gw)
        if window is None:
            return None

        return ScreenRegion(
            left=int(window.left),
            top=int(window.top),
            width=int(window.width),
            height=int(window.height),
        )

    @staticmethod
    def _find_window(candidates: Sequence[str], gw_module) -> object | None:
        seen: set[int] = set()
        for candidate in candidates:
            for window in gw_module.getWindowsWithTitle(candidate):
                if getattr(window, "isVisible", True) and id(window) not in seen:
                    seen.add(id(window))
                    return window
        all_windows = getattr(gw_module, "getAllWindows", None)
        if callable(all_windows):
            for window in all_windows():
                title = (getattr(window, "title", "") or "").lower()
                if any(candidate.lower() in title for candidate in candidates):
                    return window
        return None
