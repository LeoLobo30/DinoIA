from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from ..config import DinoVisionConfig, RealGameConfig, SimConfig, TrainingConfig
from ..project_state import find_model_candidates
from ..real_env import DinoRealEnv, RealEnvConfig


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _build_real_model(training_config: TrainingConfig, env, device: str) -> DQN:
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=training_config.learning_rate,
        gamma=training_config.gamma,
        buffer_size=training_config.buffer_size,
        learning_starts=training_config.learning_starts,
        batch_size=training_config.batch_size,
        train_freq=training_config.train_freq,
        gradient_steps=1,
        target_update_interval=training_config.target_update_interval,
        exploration_fraction=training_config.exploration_fraction,
        exploration_final_eps=training_config.exploration_final_eps,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1,
        seed=training_config.seed,
        device=device,
    )


def _load_or_create_model(
    training_config: TrainingConfig,
    env,
    device: str,
    resume: str | Path | None,
    *,
    fresh: bool,
) -> tuple[DQN, Path | None]:
    if fresh or resume in {None, "", False, "none"}:
        return _build_real_model(training_config, env, device), None

    paths: list[Path]
    if resume == "auto":
        paths = [candidate.path for candidate in find_model_candidates(training_config.output_dir)]
    else:
        paths = [Path(resume)]

    for path in paths:
        if not path.exists():
            continue
        try:
            return DQN.load(str(path), env=env, device=device), path
        except Exception:
            continue

    return _build_real_model(training_config, env, device), None


def train_real_dqn(
    *,
    training_config: TrainingConfig | None = None,
    real_config: RealGameConfig | None = None,
    vision_config: DinoVisionConfig | None = None,
    sim_config: SimConfig | None = None,
    env_config: RealEnvConfig | None = None,
    total_timesteps: int | None = None,
    output_dir: str | Path | None = None,
    device: str = "auto",
    resume: str | Path | None = "auto",
    fresh: bool = False,
) -> Path:
    training_config = training_config or TrainingConfig()
    training_config.preset = "real"
    training_config.n_envs = 1
    training_config.total_timesteps = int(total_timesteps or 5_000)
    training_config.learning_starts = min(training_config.learning_starts, max(100, training_config.total_timesteps // 10))
    training_config.buffer_size = max(training_config.buffer_size, max(20_000, training_config.total_timesteps * 4))
    training_config.exploration_fraction = 0.35
    training_config.exploration_final_eps = 0.05
    training_config.device = device
    training_config.fresh = fresh
    training_config.resume = str(resume)
    if output_dir is not None:
        training_config.output_dir = Path(output_dir)
    training_config.output_dir.mkdir(parents=True, exist_ok=True)

    real_config = real_config or RealGameConfig(debug=False)
    env_config = env_config or RealEnvConfig(step_interval_s=1.0 / max(1.0, real_config.frame_rate))
    env = Monitor(
        DinoRealEnv(
            real_config=real_config,
            vision_config=vision_config or DinoVisionConfig(),
            sim_config=sim_config or SimConfig(),
            env_config=env_config,
        )
    )
    resolved_device = _resolve_device(device)
    model, resumed_from = _load_or_create_model(
        training_config,
        env,
        resolved_device,
        resume,
        fresh=fresh,
    )
    print(f"train-real: timesteps={training_config.total_timesteps} device={resolved_device}")
    if resumed_from is not None:
        print(f"train-real: resume_from={resumed_from}")
    elif not fresh:
        print("train-real: starting fresh")

    checkpoint_dir = training_config.output_dir / "real_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100, training_config.total_timesteps // 5),
        save_path=str(checkpoint_dir),
        name_prefix="dino_real_dqn",
    )

    interrupted = False
    try:
        model.learn(
            total_timesteps=training_config.total_timesteps,
            callback=checkpoint_callback,
            progress_bar=False,
            reset_num_timesteps=resumed_from is None,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("train-real: interrupted, saving current model")
    finally:
        env.close()

    model_path = training_config.output_dir / ("dino_real_dqn_interrupted.zip" if interrupted else "dino_real_dqn_final.zip")
    model.save(model_path)
    (training_config.output_dir / "real_training_config.json").write_text(
        json.dumps(
            {
                "training": asdict(training_config),
                "real": asdict(real_config),
                "real_env": asdict(env_config),
                "resume": {
                    "requested": str(resume),
                    "fresh": fresh,
                    "resumed_from": str(resumed_from) if resumed_from is not None else None,
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return model_path
