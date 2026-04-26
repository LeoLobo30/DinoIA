from __future__ import annotations

from gymnasium.envs.registration import register

ENV_ID = "DinoSim-v0"

try:  # pragma: no cover - registration is a side effect
    register(id=ENV_ID, entry_point="dinoia.sim.env:DinoSimEnv")
except Exception:
    pass

from .env import DinoSimEnv

__all__ = ["DinoSimEnv", "ENV_ID"]
