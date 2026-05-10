from __future__ import annotations

from dataclasses import dataclass

from .types import PolicyAction


@dataclass(slots=True)
class ControlResult:
    action: PolicyAction
    executed: bool


class KeyboardController:
    def __init__(self) -> None:
        self._pyautogui = None
        self._ducking = False

    def _module(self):
        if self._pyautogui is None:
            import pyautogui

            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0
            for attr in ("MINIMUM_DURATION", "MINIMUM_SLEEP"):
                if hasattr(pyautogui, attr):
                    setattr(pyautogui, attr, 0)
            self._pyautogui = pyautogui
        return self._pyautogui

    def jump(self) -> None:
        module = self._module()
        module.keyDown("space")
        module.keyUp("space")

    def duck(self) -> None:
        self._module().keyDown("down")

    def release_duck(self) -> None:
        self._module().keyUp("down")
        self._ducking = False

    def restart(self) -> None:
        module = self._module()
        module.keyDown("space")
        module.keyUp("space")

    def perform(self, action: PolicyAction) -> ControlResult:
        if action is PolicyAction.JUMP:
            if self._ducking:
                self.release_duck()
            self.jump()
            return ControlResult(action=action, executed=True)
        if action is PolicyAction.DUCK:
            if not self._ducking:
                self.duck()
                self._ducking = True
            return ControlResult(action=action, executed=True)
        if self._ducking:
            self.release_duck()
        return ControlResult(action=action, executed=False)
