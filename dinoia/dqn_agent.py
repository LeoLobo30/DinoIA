from __future__ import annotations

import json
import shutil
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import SimConfig
from .sim.env import DinoEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DQN_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "dqn"


@dataclass(slots=True)
class DQNArtifacts:
    output_dir: Path = DEFAULT_DQN_ARTIFACTS_DIR

    @property
    def model_path(self) -> Path:
        return self.output_dir / "sim_model.zip"

    @property
    def best_model_path(self) -> Path:
        return self.output_dir / "best_sim_model.zip"

    @property
    def training_config_path(self) -> Path:
        return self.output_dir / "training_config.json"

    @property
    def training_history_path(self) -> Path:
        return self.output_dir / "training_history.json"

    @property
    def evaluation_history_path(self) -> Path:
        return self.output_dir / "evaluation_history.json"

    @property
    def latest_eval_path(self) -> Path:
        return self.output_dir / "latest_eval.json"

    @property
    def best_eval_path(self) -> Path:
        return self.output_dir / "best_eval.json"

    @property
    def latest_summary_path(self) -> Path:
        return self.output_dir / "latest_summary.txt"

    @property
    def runs_root(self) -> Path:
        return self.output_dir / "runs"

    def run_dir(self, stage: str) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        return self.runs_root / f"{stamp}-{stage}-{uuid.uuid4().hex[:8]}"


@dataclass(slots=True)
class DQNTrainResult:
    model_path: Path
    best_model_path: Path | None
    run_dir: Path
    total_timesteps: int
    config_path: Path
    history_path: Path
    evaluation_path: Path


@dataclass(slots=True)
class DQNEvalResult:
    model_path: Path
    run_dir: Path
    episodes: int
    steps: int
    mean_reward: float
    reward_std: float
    mean_passed_obstacles: float
    max_passed_obstacles: int
    action_counts: dict[str, int]
    episode_rewards: list[float]
    episode_lengths: list[int]


@dataclass(slots=True)
class DQNRunResult:
    model_path: Path
    steps: int
    reward: float
    passed_obstacles: int
    episodes: int
    action_counts: dict[str, int]


@dataclass(slots=True)
class DQNProjectResults:
    output_dir: Path
    model_path: Path | None
    best_model_path: Path | None
    training_config_path: Path | None
    history_path: Path | None
    evaluation_history_path: Path | None
    latest_eval_path: Path | None
    best_eval_path: Path | None
    summary_path: Path | None
    history: list[dict[str, Any]]
    evaluation_history: list[dict[str, Any]]

    @property
    def selected_model_path(self) -> Path | None:
        return self.best_model_path or self.model_path

    @property
    def latest_total_timesteps(self) -> int | None:
        if not self.history:
            return None
        value = self.history[-1].get("total_timesteps")
        return int(value) if value is not None else None

    @property
    def latest_eval(self) -> dict[str, Any] | None:
        return self.evaluation_history[-1] if self.evaluation_history else None

    @property
    def best_eval(self) -> dict[str, Any] | None:
        if self.best_eval_path is None:
            return None
        raw = _read_json(self.best_eval_path)
        return raw if isinstance(raw, dict) else None


def train_sim_dqn(
    *,
    total_timesteps: int = 300_000,
    seed: int = 42,
    learning_rate: float = 1e-4,
    buffer_size: int = 200_000,
    learning_starts: int = 5_000,
    batch_size: int = 64,
    n_quantiles: int = 50,
    gamma: float = 0.99,
    n_envs: int = 1,
    train_freq: int = 4,
    gradient_steps: int = 1,
    target_update_interval: int = 10_000,
    exploration_initial_eps: float = 1.0,
    exploration_fraction: float = 0.25,
    exploration_final_eps: float = 0.05,
    eval_freq: int = 10_000,
    eval_episodes: int = 20,
    checkpoint_freq: int = 50_000,
    device: str = "auto",
    output_dir: str | Path | None = None,
    sim_config: SimConfig | None = None,
    source_model: str | Path | None = None,
    verbose: int = 1,
) -> DQNTrainResult:
    total_timesteps = _positive_int("total_timesteps", total_timesteps)
    artifacts = _ensure_artifacts(output_dir)
    run_dir = _make_run_dir(artifacts, "train")
    n_envs = _positive_int("n_envs", n_envs)
    sim_config = sim_config or SimConfig(seed=seed)
    eval_config = _clean_eval_config(sim_config)
    device = resolve_torch_device(device)
    QRDQN = _load_qrdqn()
    Monitor, DummyVecEnv, CallbackList, CheckpointCallback, EvalCallback = _load_sb3_training_helpers()
    env = DummyVecEnv(
        [
            _make_monitored_env_factory(sim_config=sim_config, seed=seed, env_index=i, Monitor=Monitor)
            for i in range(n_envs)
        ]
    )
    eval_env = Monitor(DinoEnv(config=eval_config))
    run_best_model_path = run_dir / "best" / "best_model.zip"
    hyperparameters = {
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "batch_size": batch_size,
        "n_quantiles": n_quantiles,
        "gamma": gamma,
        "n_envs": n_envs,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "target_update_interval": target_update_interval,
        "exploration_initial_eps": exploration_initial_eps,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "eval_freq": eval_freq,
        "eval_episodes": eval_episodes,
        "checkpoint_freq": checkpoint_freq,
        "device": device,
        "seed": seed,
    }
    dqn_hyperparameters = _dqn_init_hyperparameters(hyperparameters)
    source_path = Path(source_model) if source_model is not None else None
    if source_path is not None:
        model = QRDQN.load(
            str(source_path),
            env=env,
            device=device,
            **{key: value for key, value in dqn_hyperparameters.items() if key != "device"},
        )
    else:
        model = QRDQN(
            "MlpPolicy",
            env,
            verbose=verbose,
            policy_kwargs={"n_quantiles": n_quantiles},
            **dqn_hyperparameters,
        )

    try:
        callback = _build_training_callback(
            CallbackList=CallbackList,
            CheckpointCallback=CheckpointCallback,
            EvalCallback=EvalCallback,
            eval_env=eval_env,
            run_dir=run_dir,
            eval_freq=eval_freq,
            eval_episodes=eval_episodes,
            checkpoint_freq=checkpoint_freq,
        )
        model.learn(total_timesteps=total_timesteps, callback=callback)
        model.save(str(artifacts.model_path))
        if run_best_model_path.exists():
            shutil.copy2(run_best_model_path, artifacts.best_model_path)
    finally:
        env.close()
        eval_env.close()

    config_doc = {
        "algorithm": "QR-DQN",
        "total_timesteps": total_timesteps,
        "seed": seed,
        "device": device,
        "source_model": source_path,
        "hyperparameters": hyperparameters,
        "effective_hyperparameters": _effective_dqn_hyperparameters(model),
        "sim_config": sim_config,
        "eval_config": eval_config,
        "model_path": artifacts.model_path,
        "run_best_model_path": run_best_model_path if run_best_model_path.exists() else None,
    }
    _write_json(run_dir / "training_config.json", config_doc)
    _write_json(artifacts.training_config_path, config_doc)

    eval_result = evaluate_sim_dqn(
        model_path=artifacts.best_model_path if run_best_model_path.exists() else artifacts.model_path,
        episodes=eval_episodes,
        seed=seed,
        device=device,
        output_dir=artifacts.output_dir,
        sim_config=eval_config,
    )
    _update_best_model(artifacts, eval_result)
    entry = {
        "stage": "sim",
        "algorithm": "QR-DQN",
        "created_at": _utc_now(),
        "total_timesteps": total_timesteps,
        "model_path": artifacts.model_path,
        "best_model_path": artifacts.best_model_path
        if artifacts.best_model_path.exists()
        else None,
        "config_path": artifacts.training_config_path,
        "evaluation_path": artifacts.latest_eval_path,
        "mean_reward": eval_result.mean_reward,
        "mean_passed_obstacles": eval_result.mean_passed_obstacles,
    }
    _append_history(artifacts.training_history_path, entry)
    _write_latest_summary(artifacts, entry)
    return DQNTrainResult(
        model_path=artifacts.model_path,
        best_model_path=artifacts.best_model_path
        if artifacts.best_model_path.exists()
        else None,
        run_dir=run_dir,
        total_timesteps=total_timesteps,
        config_path=artifacts.training_config_path,
        history_path=artifacts.training_history_path,
        evaluation_path=artifacts.latest_eval_path,
    )


def evaluate_sim_dqn(
    *,
    model_path: str | Path | None = None,
    best: bool = False,
    episodes: int = 10,
    seed: int = 42,
    device: str = "auto",
    output_dir: str | Path | None = None,
    sim_config: SimConfig | None = None,
) -> DQNEvalResult:
    artifacts = _ensure_artifacts(output_dir)
    selected_model_path = _select_model_path(artifacts, model_path, best=best)
    QRDQN = _load_qrdqn()
    device = resolve_torch_device(device)
    env = DinoEnv(config=sim_config or SimConfig(seed=seed))
    model = QRDQN.load(str(selected_model_path), device=device)
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    passed_obstacles: list[int] = []
    action_counts: Counter[str] = Counter()
    steps = 0

    try:
        for episode in range(_positive_int("episodes", episodes)):
            obs, info = env.reset(seed=seed + episode)
            done = False
            episode_reward = 0.0
            episode_length = 0
            last_info = info
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_int = _action_to_int(action)
                action_counts[_action_name(action_int)] += 1
                obs, reward, terminated, truncated, last_info = env.step(action_int)
                episode_reward += float(reward)
                episode_length += 1
                steps += 1
                done = bool(terminated or truncated)
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            passed_obstacles.append(int(last_info.get("passed_obstacles", 0)))
    finally:
        env.close()

    rewards = np.asarray(episode_rewards, dtype=np.float32)
    run_dir = _make_run_dir(artifacts, "eval")
    metrics = {
        "algorithm": "QR-DQN",
        "created_at": _utc_now(),
        "model_path": selected_model_path,
        "episodes": episodes,
        "steps": steps,
        "mean_reward": float(np.mean(rewards)) if rewards.size else 0.0,
        "reward_std": float(np.std(rewards)) if rewards.size else 0.0,
        "mean_passed_obstacles": float(np.mean(passed_obstacles))
        if passed_obstacles
        else 0.0,
        "max_passed_obstacles": int(max(passed_obstacles, default=0)),
        "action_counts": dict(action_counts),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }
    _write_json(run_dir / "eval.json", metrics)
    _write_json(artifacts.latest_eval_path, metrics)
    _append_history(artifacts.evaluation_history_path, metrics)
    return DQNEvalResult(
        model_path=selected_model_path,
        run_dir=run_dir,
        episodes=episodes,
        steps=steps,
        mean_reward=float(metrics["mean_reward"]),
        reward_std=float(metrics["reward_std"]),
        mean_passed_obstacles=float(metrics["mean_passed_obstacles"]),
        max_passed_obstacles=int(metrics["max_passed_obstacles"]),
        action_counts=dict(action_counts),
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
    )


def play_sim_dqn(
    *,
    model_path: str | Path | None = None,
    best: bool = False,
    duration: float = 30.0,
    output_dir: str | Path | None = None,
    device: str = "auto",
    no_display: bool = False,
    sim_config: SimConfig | None = None,
) -> DQNRunResult:
    artifacts = _ensure_artifacts(output_dir)
    selected_model_path = _select_model_path(artifacts, model_path, best=best)
    QRDQN = _load_qrdqn()
    device = resolve_torch_device(device)
    env = DinoEnv(
        config=sim_config or SimConfig(), render_mode=None if no_display else "human"
    )
    model = QRDQN.load(str(selected_model_path), device=device)
    obs, _ = env.reset()
    deadline = time.time() + max(0.0, float(duration))
    reward_total = 0.0
    steps = 0
    episodes = 1
    passed = 0
    action_counts: Counter[str] = Counter()

    try:
        while time.time() < deadline:
            action, _ = model.predict(obs, deterministic=True)
            action_int = _action_to_int(action)
            action_counts[_action_name(action_int)] += 1
            obs, reward, terminated, truncated, info = env.step(action_int)
            reward_total += float(reward)
            steps += 1
            passed = max(passed, int(info.get("passed_obstacles", 0)))
            if not no_display:
                env.render()
            if terminated or truncated:
                obs, _ = env.reset()
                episodes += 1
    finally:
        env.close()

    return DQNRunResult(
        model_path=selected_model_path,
        steps=steps,
        reward=reward_total,
        passed_obstacles=passed,
        episodes=episodes,
        action_counts=dict(action_counts),
    )


def summarize_dqn(output_dir: str | Path | None = None) -> DQNProjectResults:
    artifacts = _artifacts(output_dir)
    return DQNProjectResults(
        output_dir=artifacts.output_dir,
        model_path=_existing(artifacts.model_path),
        best_model_path=_existing(artifacts.best_model_path),
        training_config_path=_existing(artifacts.training_config_path),
        history_path=_existing(artifacts.training_history_path),
        evaluation_history_path=_existing(artifacts.evaluation_history_path),
        latest_eval_path=_existing(artifacts.latest_eval_path),
        best_eval_path=_existing(artifacts.best_eval_path),
        summary_path=_existing(artifacts.latest_summary_path),
        history=_read_history(artifacts.training_history_path),
        evaluation_history=_read_history(artifacts.evaluation_history_path),
    )


def resolve_torch_device(requested_device: str = "auto") -> str:
    requested = (requested_device or "auto").strip().lower()
    if requested in {"gpu", "cuda:0"}:
        requested = "cuda"
    if requested == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if requested.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but PyTorch does not see a CUDA device."
            )
    if requested != "cpu" and not requested.startswith("cuda"):
        raise ValueError(f"Unsupported device: {requested_device!r}")
    return requested


def _load_qrdqn():
    try:
        from sb3_contrib import QRDQN
    except Exception as exc:
        raise RuntimeError(
            "sb3-contrib is required for QR-DQN. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return QRDQN


def _load_sb3_training_helpers():
    try:
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as exc:
        raise RuntimeError(
            "stable-baselines3 training helpers are required. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return Monitor, DummyVecEnv, CallbackList, CheckpointCallback, EvalCallback


def _make_monitored_env_factory(*, sim_config: SimConfig, seed: int, env_index: int, Monitor: Any):
    def _factory():
        env = DinoEnv(config=SimConfig(**asdict(sim_config)))
        env.reset(seed=seed + env_index)
        return Monitor(env)

    return _factory


def _clean_eval_config(sim_config: SimConfig) -> SimConfig:
    data = asdict(sim_config)
    data["domain_randomization"] = False
    data["observation_noise"] = 0.0
    data["action_latency_steps"] = 0
    data["observation_latency_steps"] = 0
    data["curriculum_learning"] = False
    return SimConfig(**data)


def _dqn_init_hyperparameters(hyperparameters: dict[str, Any]) -> dict[str, Any]:
    excluded = {"eval_freq", "eval_episodes", "checkpoint_freq", "n_quantiles", "n_envs"}
    return {key: value for key, value in hyperparameters.items() if key not in excluded}


def _build_training_callback(
    *,
    CallbackList: Any,
    CheckpointCallback: Any,
    EvalCallback: Any,
    eval_env: Any,
    run_dir: Path,
    eval_freq: int,
    eval_episodes: int,
    checkpoint_freq: int,
) -> Any | None:
    callbacks = []
    if eval_freq > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir / "best"),
                log_path=str(run_dir / "eval"),
                eval_freq=eval_freq,
                n_eval_episodes=eval_episodes,
                deterministic=True,
                render=False,
                verbose=1,
            )
        )
    if checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(run_dir / "checkpoints"),
                name_prefix="dqn_checkpoint",
                save_replay_buffer=True,
            )
        )
    if not callbacks:
        return None
    return CallbackList(callbacks)


def _effective_dqn_hyperparameters(model: Any) -> dict[str, Any]:
    keys = (
        "learning_rate",
        "buffer_size",
        "learning_starts",
        "batch_size",
        "n_quantiles",
        "gradient_steps",
        "gamma",
        "train_freq",
        "target_update_interval",
        "exploration_initial_eps",
        "exploration_fraction",
        "exploration_final_eps",
        "device",
    )
    return {
        key: _json_ready(getattr(model, key)) for key in keys if hasattr(model, key)
    }


def _artifacts(output_dir: str | Path | None = None) -> DQNArtifacts:
    return DQNArtifacts(
        Path(output_dir) if output_dir is not None else DEFAULT_DQN_ARTIFACTS_DIR
    )


def _ensure_artifacts(output_dir: str | Path | None = None) -> DQNArtifacts:
    artifacts = _artifacts(output_dir)
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts.runs_root.mkdir(parents=True, exist_ok=True)
    return artifacts


def _select_model_path(
    artifacts: DQNArtifacts, model_path: str | Path | None, *, best: bool
) -> Path:
    if model_path is not None:
        path = Path(model_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"QR-DQN model not found: {path}")
    candidates = (
        (artifacts.best_model_path, artifacts.model_path)
        if best
        else (artifacts.model_path, artifacts.best_model_path)
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No QR-DQN model found in {artifacts.output_dir}. Run train-sim-dqn first."
    )


def _update_best_model(artifacts: DQNArtifacts, eval_result: DQNEvalResult) -> None:
    current_best = _read_json(artifacts.best_eval_path)
    current_reward = (
        float(current_best.get("mean_reward", float("-inf")))
        if isinstance(current_best, dict)
        else float("-inf")
    )
    if eval_result.mean_reward < current_reward:
        return
    if eval_result.model_path.resolve() != artifacts.best_model_path.resolve():
        shutil.copy2(eval_result.model_path, artifacts.best_model_path)
    _write_json(artifacts.best_eval_path, asdict(eval_result))


def _make_run_dir(artifacts: DQNArtifacts, stage: str) -> Path:
    path = artifacts.run_dir(stage)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_latest_summary(artifacts: DQNArtifacts, entry: dict[str, Any]) -> None:
    lines = [
        "DinoIA QR-DQN summary",
        f"model_path={entry.get('model_path')}",
        f"best_model_path={entry.get('best_model_path') or 'none'}",
        f"total_timesteps={entry.get('total_timesteps')}",
        f"mean_reward={entry.get('mean_reward')}",
        f"mean_passed_obstacles={entry.get('mean_passed_obstacles')}",
    ]
    artifacts.latest_summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_history(path: Path, entry: dict[str, Any]) -> None:
    history = _read_history(path)
    history.append(_json_ready(entry))
    _write_json(path, history)


def _read_history(path: Path) -> list[dict[str, Any]]:
    raw = _read_json(path)
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(data), indent=2, sort_keys=True), encoding="utf-8"
    )


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _existing(path: Path) -> Path | None:
    return path if path.exists() else None


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _action_to_int(action: Any) -> int:
    return int(np.asarray(action).reshape(-1)[0])


def _action_name(action: int) -> str:
    return {0: "noop", 1: "jump", 2: "duck"}.get(int(action), str(action))


def load_sim_config_from_artifacts(
    output_dir: str | Path | None = None,
) -> SimConfig | None:
    artifacts = _artifacts(output_dir)
    raw = _read_json(artifacts.training_config_path)
    if not isinstance(raw, dict) or not isinstance(raw.get("sim_config"), dict):
        return None
    allowed = {field.name for field in fields(SimConfig)}
    return SimConfig(
        **{key: value for key, value in raw["sim_config"].items() if key in allowed}
    )


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
