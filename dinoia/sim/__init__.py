from __future__ import annotations

from gymnasium.envs.registration import register

ENV_ID = "Dino-v0"
LEGACY_ENV_ID = "DinoSim-v0"

try:  # pragma: no cover - registration is a side effect
    register(id=ENV_ID, entry_point="dinoia.sim.env:DinoEnv")
except Exception:
    pass

try:  # pragma: no cover - registration is a side effect
    register(id=LEGACY_ENV_ID, entry_point="dinoia.sim.env:DinoSimEnv")
except Exception:
    pass

from .env import DinoEnv, DinoSimEnv

__all__ = ["DinoEnv", "DinoSimEnv", "ENV_ID", "LEGACY_ENV_ID"]
