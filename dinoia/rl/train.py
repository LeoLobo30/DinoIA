from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from ..config import SimConfig, TrainingConfig
from ..project_state import find_model_candidates
from ..sim.env import DinoSimEnv

TRAINING_PRESETS: dict[str, dict[str, object]] = {
    "quick": {
        "total_timesteps": 20_000,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "buffer_size": 50_000,
        "learning_starts": 1_000,
        "batch_size": 64,
        "target_update_interval": 1_000,
        "exploration_fraction": 0.45,
        "exploration_final_eps": 0.08,
        "n_eval_episodes": 3,
        "n_envs": 2,
    },
    "standard": {
        "total_timesteps": 120_000,
        "learning_rate": 3e-4,
        "gamma": 0.995,
        "buffer_size": 200_000,
        "learning_starts": 5_000,
        "batch_size": 128,
        "target_update_interval": 2_500,
        "exploration_fraction": 0.35,
        "exploration_final_eps": 0.02,
        "n_eval_episodes": 10,
        "n_envs": 4,
    },
    "long": {
        "total_timesteps": 300_000,
        "learning_rate": 3e-4,
        "gamma": 0.995,
        "buffer_size": 300_000,
        "learning_starts": 8_000,
        "batch_size": 128,
        "target_update_interval": 4_000,
        "exploration_fraction": 0.28,
        "exploration_final_eps": 0.01,
        "n_eval_episodes": 12,
        "n_envs": 4,
    },
    "real-short": {
        "total_timesteps": 30_000,
        "learning_rate": 3e-4,
        "gamma": 0.995,
        "buffer_size": 150_000,
        "learning_starts": 3_000,
        "batch_size": 128,
        "target_update_interval": 2_000,
        "exploration_fraction": 0.4,
        "exploration_final_eps": 0.04,
        "n_eval_episodes": 8,
        "n_envs": 4,
    },
}


class StopOnEventCallback(BaseCallback):
    def __init__(self, stop_event: threading.Event):
        super().__init__()
        self.stop_event = stop_event

    def _on_step(self) -> bool:  # pragma: no cover - stable-baselines3 hook
        return not self.stop_event.is_set()


def _build_vec_env(sim_config: SimConfig, *, seed: int, n_envs: int = 1, render_mode: str | None = None):
    def _make_env(rank: int):
        def _init():
            env = DinoSimEnv(config=sim_config, render_mode=render_mode)
            return Monitor(env)

        return _init

    env_fns = [_make_env(index) for index in range(max(1, n_envs))]
    if len(env_fns) == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)
    env.seed(seed)
    return env


def _gap_range(min_gap: int, max_gap: int) -> tuple[int, int]:
    min_gap = max(60, int(min_gap))
    max_gap = max(min_gap + 20, int(max_gap))
    return min_gap, max_gap


def _build_curriculum(sim_config: SimConfig, total_timesteps: int) -> list[tuple[str, int, SimConfig]]:
    easy_min_gap, easy_max_gap = _gap_range(
        int(sim_config.min_spawn_gap_px * 1.2),
        int(sim_config.max_spawn_gap_px * 1.25),
    )
    hard_min_gap, hard_max_gap = _gap_range(
        max(160, int(sim_config.min_spawn_gap_px * 0.85)),
        max(240, int(sim_config.max_spawn_gap_px * 0.9)),
    )

    easy_config = replace(
        sim_config,
        base_speed=max(240.0, sim_config.base_speed * 0.82),
        speed_increment=max(4.5, sim_config.speed_increment * 0.8),
        min_spawn_gap_px=easy_min_gap,
        max_spawn_gap_px=easy_max_gap,
        bird_probability=max(0.05, sim_config.bird_probability * 0.6),
    )
    medium_config = replace(sim_config)
    hard_config = replace(
        sim_config,
        base_speed=min(sim_config.max_speed - 40.0, sim_config.base_speed * 1.12),
        speed_increment=sim_config.speed_increment * 1.18,
        min_spawn_gap_px=hard_min_gap,
        max_spawn_gap_px=hard_max_gap,
        bird_probability=min(0.35, sim_config.bird_probability * 1.35),
    )

    stage_specs = [
        ("easy", 0.35, easy_config),
        ("medium", 0.35, medium_config),
        ("hard", 0.30, hard_config),
    ]

    stages: list[tuple[str, int, SimConfig]] = []
    assigned = 0
    for name, ratio, stage_config in stage_specs:
        stage_timesteps = int(total_timesteps * ratio)
        stages.append((name, stage_timesteps, stage_config))
        assigned += stage_timesteps

    if stages:
        stages[-1] = (stages[-1][0], stages[-1][1] + max(0, total_timesteps - assigned), stages[-1][2])
    return stages


def _build_callbacks(
    training_config: TrainingConfig,
    stage_name: str,
    stage_timesteps: int,
    eval_env,
    stop_event: threading.Event | None = None,
) -> CallbackList:
    checkpoint_dir = training_config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = training_config.output_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    save_freq = max(1, stage_timesteps // max(1, training_config.n_envs * 4))
    eval_freq = max(1, stage_timesteps // max(1, training_config.n_envs * 4))
    callbacks = [
        CheckpointCallback(
            save_freq=save_freq,
            save_path=str(checkpoint_dir),
            name_prefix=f"dino_dqn_{stage_name}",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(training_config.output_dir / "eval"),
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=training_config.n_eval_episodes,
        ),
    ]
    if stop_event is not None:
        callbacks.append(StopOnEventCallback(stop_event))
    return CallbackList(callbacks)


def _apply_preset(training_config: TrainingConfig) -> TrainingConfig:
    preset = (training_config.preset or "standard").lower()
    preset_config = TRAINING_PRESETS.get(preset, TRAINING_PRESETS["standard"])
    training_config.preset = preset if preset in TRAINING_PRESETS else "standard"

    for field_name, value in preset_config.items():
        setattr(training_config, field_name, value)

    return training_config


def _build_model(training_config: TrainingConfig, train_env, resolved_device: str) -> DQN:
    policy_kwargs = dict(net_arch=[128, 128])
    return DQN(
        "MlpPolicy",
        train_env,
        learning_rate=training_config.learning_rate,
        gamma=training_config.gamma,
        buffer_size=training_config.buffer_size,
        learning_starts=training_config.learning_starts,
        batch_size=training_config.batch_size,
        train_freq=training_config.train_freq,
        gradient_steps=1,
        replay_buffer_kwargs={"handle_timeout_termination": False},
        target_update_interval=training_config.target_update_interval,
        exploration_fraction=training_config.exploration_fraction,
        exploration_final_eps=training_config.exploration_final_eps,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=training_config.seed,
        device=resolved_device,
    )


def _build_or_resume_model(
    training_config: TrainingConfig,
    train_env,
    resolved_device: str,
    resume: str | Path | None,
    *,
    fresh: bool,
) -> tuple[DQN, Path | None, list[str]]:
    if fresh or resume in {None, "", False, "none"}:
        return _build_model(training_config, train_env, resolved_device), None, []

    expected_obs_size = training_config.observation_size
    notes: list[str] = []
    candidate_paths: list[Path]

    if resume == "auto":
        candidate_paths = [candidate.path for candidate in find_model_candidates(training_config.output_dir)]
        if not candidate_paths:
            notes.append("no checkpoints found for auto resume")
    else:
        candidate_paths = [Path(resume)]

    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            notes.append(f"skip {candidate_path}: file not found")
            continue
        try:
            model = DQN.load(str(candidate_path), env=train_env, device=resolved_device)
            return model, candidate_path, notes
        except Exception as exc:
            notes.append(f"skip {candidate_path.name}: {exc.__class__.__name__}: {exc}")
            continue

    return _build_model(training_config, train_env, resolved_device), None, notes


def train_dqn(
    training_config: TrainingConfig | None = None,
    sim_config: SimConfig | None = None,
    *,
    preset: str | None = None,
    total_timesteps: int | None = None,
    output_dir: str | Path | None = None,
    n_envs: int | None = None,
    stop_event: threading.Event | None = None,
    device: str | None = None,
    resume: str | Path | None = None,
    fresh: bool = False,
):
    training_config = training_config or TrainingConfig()
    if preset is not None:
        training_config.preset = preset
    training_config = _apply_preset(training_config)

    if total_timesteps is not None:
        training_config.total_timesteps = total_timesteps
    if output_dir is not None:
        training_config.output_dir = Path(output_dir)
    if n_envs is not None:
        training_config.n_envs = max(1, n_envs)
    if resume is not None:
        training_config.resume = str(resume)
    training_config.fresh = fresh

    training_config.output_dir.mkdir(parents=True, exist_ok=True)

    sim_config = sim_config or SimConfig(seed=training_config.seed)
    resolved_device = device or training_config.device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        resolved_device = "cpu"

    stages = _build_curriculum(sim_config, training_config.total_timesteps)
    if not stages:
        stages = [("single", training_config.total_timesteps, sim_config)]

    _, _, first_stage_config = stages[0]
    train_stage_config = replace(first_stage_config, domain_randomization=True)
    eval_stage_config = replace(first_stage_config, domain_randomization=False, observation_noise=0.0)
    train_env = _build_vec_env(train_stage_config, seed=training_config.seed, n_envs=max(1, training_config.n_envs))
    eval_env = _build_vec_env(replace(eval_stage_config, seed=training_config.seed + 1), seed=training_config.seed + 1, n_envs=1)
    model, resumed_from, resume_notes = _build_or_resume_model(
        training_config,
        train_env,
        resolved_device,
        resume if resume is not None else training_config.resume,
        fresh=fresh or training_config.fresh,
    )
    cuda_status = "available" if torch.cuda.is_available() else "unavailable"
    print(f"train: preset={training_config.preset} device={resolved_device} cuda={cuda_status}")
    if resumed_from is not None:
        print(f"train: resume_from={resumed_from}")
    elif resume_notes:
        for note in resume_notes:
            print(f"train: {note}")
        print("train: starting fresh")

    try:
        total_learned = 0
        for stage_index, (stage_name, stage_timesteps, stage_config) in enumerate(stages, start=1):
            if stage_index > 1:
                train_env.close()
                eval_env.close()
                train_stage_config = replace(stage_config, domain_randomization=True)
                eval_stage_config = replace(stage_config, domain_randomization=False, observation_noise=0.0)
                train_env = _build_vec_env(train_stage_config, seed=training_config.seed + stage_index, n_envs=max(1, training_config.n_envs))
                eval_env = _build_vec_env(replace(eval_stage_config, seed=training_config.seed + stage_index + 1), seed=training_config.seed + stage_index + 1, n_envs=1)
                model.set_env(train_env)

            print(
                f"train: stage={stage_index}/{len(stages)} name={stage_name} timesteps={stage_timesteps} "
                f"base_speed={stage_config.base_speed:.1f} spawn_gap={stage_config.min_spawn_gap_px}-{stage_config.max_spawn_gap_px}"
            )
            callbacks = _build_callbacks(training_config, stage_name, stage_timesteps, eval_env, stop_event=stop_event)

            model.learn(
                total_timesteps=stage_timesteps,
                callback=callbacks,
                progress_bar=False,
                reset_num_timesteps=False if total_learned or resumed_from is not None else True,
            )
            total_learned += stage_timesteps
    finally:
        train_env.close()
        eval_env.close()

    model_path = training_config.output_dir / "dino_dqn_final.zip"
    model.save(model_path)

    (training_config.output_dir / "training_config.json").write_text(
        json.dumps(
            {
                "training": asdict(training_config),
                "simulation": asdict(sim_config),
                "curriculum": [
                    {
                        "name": name,
                        "timesteps": timesteps,
                        "simulation": asdict(stage_config),
                    }
                    for name, timesteps, stage_config in stages
                ],
                "resume": {
                    "requested": str(resume if resume is not None else training_config.resume),
                    "fresh": fresh or training_config.fresh,
                    "resumed_from": str(resumed_from) if resumed_from is not None else None,
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return model_path




