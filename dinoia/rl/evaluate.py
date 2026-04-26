from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import DQN

from ..config import SimConfig
from ..project_state import select_best_model, select_latest_model
from ..sim.env import DinoSimEnv


def resolve_model_path(
    model_path: str | Path | None = None,
    *,
    latest: bool = False,
    best: bool = False,
) -> Path:
    if model_path is not None:
        return Path(model_path)
    if best:
        selected = select_best_model()
    else:
        selected = select_latest_model() if latest or not best else select_best_model()
    if selected is None:
        raise FileNotFoundError("No trained model was found in artifacts/dqn. Run train first.")
    return selected


def evaluate_model(
    model_path: str | Path | None = None,
    *,
    episodes: int = 5,
    render: bool = False,
    seed: int = 42,
    latest: bool = False,
    best: bool = False,
) -> dict[str, Any]:
    resolved_model_path = resolve_model_path(model_path, latest=latest, best=best)
    env = DinoSimEnv(
        config=SimConfig(seed=seed, domain_randomization=False, observation_noise=0.0),
        render_mode="human" if render else None,
    )
    model = DQN.load(str(resolved_model_path), env=env)

    rewards: list[float] = []
    lengths: list[int] = []

    for episode in range(episodes):
        obs, _info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        total_reward = 0.0
        length = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(int(action))
            total_reward += float(reward)
            length += 1
            if render:
                env.render()

        rewards.append(total_reward)
        lengths.append(length)

    env.close()
    summary = {
        "model_path": str(resolved_model_path),
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "std_length": float(np.std(lengths)) if lengths else 0.0,
        "rewards": rewards,
        "lengths": lengths,
    }
    return summary
