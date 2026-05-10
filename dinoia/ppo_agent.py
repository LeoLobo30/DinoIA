from __future__ import annotations

import json
import shutil
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .config import DinoVisionConfig, RealGameConfig, SimConfig
from .real_env import DinoRealEnv, RealEnvConfig
from .sim.env import DinoEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PPO_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "ppo"


@dataclass(slots=True)
class PPOArtifacts:
    output_dir: Path = DEFAULT_PPO_ARTIFACTS_DIR

    @property
    def sim_model_path(self) -> Path:
        return self.output_dir / "sim_model.zip"

    @property
    def best_sim_model_path(self) -> Path:
        return self.output_dir / "best_sim_model.zip"

    @property
    def real_model_path(self) -> Path:
        return self.output_dir / "real_model.zip"

    @property
    def best_real_model_path(self) -> Path:
        return self.output_dir / "best_real_model.zip"

    @property
    def best_model_path(self) -> Path:
        return self.output_dir / "best_model.zip"

    @property
    def sim_vecnormalize_path(self) -> Path:
        return self.output_dir / "sim_vecnormalize.pkl"

    @property
    def best_sim_vecnormalize_path(self) -> Path:
        return self.output_dir / "best_sim_vecnormalize.pkl"

    @property
    def real_vecnormalize_path(self) -> Path:
        return self.output_dir / "real_vecnormalize.pkl"

    @property
    def best_real_vecnormalize_path(self) -> Path:
        return self.output_dir / "best_real_vecnormalize.pkl"

    @property
    def best_model_vecnormalize_path(self) -> Path:
        return self.output_dir / "best_model.vecnormalize.pkl"

    @property
    def sim_training_config_path(self) -> Path:
        return self.output_dir / "sim_training_config.json"

    @property
    def real_training_config_path(self) -> Path:
        return self.output_dir / "real_training_config.json"

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
    def best_sim_eval_path(self) -> Path:
        return self.output_dir / "best_sim_eval.json"

    @property
    def best_real_eval_path(self) -> Path:
        return self.output_dir / "best_real_eval.json"

    @property
    def best_metrics_path(self) -> Path:
        return self.output_dir / "best_metrics.json"

    @property
    def latest_summary_path(self) -> Path:
        return self.output_dir / "latest_summary.txt"

    @property
    def runs_root(self) -> Path:
        return self.output_dir / "runs"

    def run_dir(self, stage: str) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        suffix = uuid.uuid4().hex[:8]
        return self.runs_root / f"{stamp}-{stage}-{suffix}"


@dataclass(slots=True)
class PPOTrainResult:
    stage: str
    model_path: Path
    best_model_path: Path | None
    run_dir: Path
    total_timesteps: int
    config_path: Path
    history_path: Path
    evaluation_path: Path | None = None
    source_model_path: Path | None = None


@dataclass(slots=True)
class PPORunResult:
    mode: str
    model_path: Path
    steps: int
    reward: float
    passed_obstacles: int
    episodes: int
    action_counts: dict[str, int] | None = None
    noops: int = 0
    jumps: int = 0
    ducks: int = 0


@dataclass(slots=True)
class PPOEvalResult:
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
class PPOProjectResults:
    output_dir: Path
    sim_model_path: Path | None
    best_sim_model_path: Path | None
    real_model_path: Path | None
    best_real_model_path: Path | None
    sim_vecnormalize_path: Path | None
    best_sim_vecnormalize_path: Path | None
    real_vecnormalize_path: Path | None
    best_real_vecnormalize_path: Path | None
    best_model_vecnormalize_path: Path | None
    sim_training_config_path: Path | None
    real_training_config_path: Path | None
    history_path: Path | None
    evaluation_history_path: Path | None
    latest_eval_path: Path | None
    best_sim_eval_path: Path | None
    best_real_eval_path: Path | None
    best_metrics_path: Path | None
    summary_path: Path | None
    history: list[dict[str, Any]]
    evaluation_history: list[dict[str, Any]]

    @property
    def best_model_path(self) -> Path | None:
        if self.best_metrics_path and self.best_metrics_path.exists():
            raw = _read_json(self.best_metrics_path)
            if isinstance(raw, dict):
                model_path = raw.get("model_path")
                if model_path:
                    path = Path(model_path)
                    if path.exists():
                        return path
        return self.best_real_model_path or self.best_sim_model_path or self.real_model_path or self.sim_model_path

    @property
    def latest_stage(self) -> str | None:
        if not self.history:
            return None
        stage = self.history[-1].get("stage")
        return str(stage) if stage is not None else None

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
        if not self.best_metrics_path or not self.best_metrics_path.exists():
            return None
        raw = _read_json(self.best_metrics_path)
        return raw if isinstance(raw, dict) else None

    @property
    def best_eval_mean_reward(self) -> float | None:
        metrics = self.best_eval
        if metrics is None:
            return None
        value = metrics.get("mean_reward")
        return float(value) if value is not None else None

    @property
    def latest_eval_mean_reward(self) -> float | None:
        metrics = self.latest_eval
        if metrics is None:
            return None
        value = metrics.get("mean_reward")
        return float(value) if value is not None else None


def _load_ppo():
    try:
        from stable_baselines3 import PPO
    except Exception as exc:  # pragma: no cover - exercised when dependency is missing
        raise RuntimeError(
            "stable-baselines3 is required for PPO. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return PPO


def _copy_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _vecnormalize_companion_path(model_path: Path) -> Path:
    if model_path.suffix:
        return model_path.with_suffix(".vecnormalize.pkl")
    return model_path.with_name(f"{model_path.name}.vecnormalize.pkl")


def _save_vecnormalize(vec_env: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vec_env.save(str(path))


def _load_vecnormalize_stats(path: Path, vec_env: Any, *, training: bool, norm_reward: bool = False) -> Any:
    if path.exists():
        loaded = VecNormalize.load(str(path), vec_env)
        loaded.training = training
        loaded.norm_reward = norm_reward
        return loaded
    return VecNormalize(vec_env, training=training, norm_obs=True, norm_reward=norm_reward, clip_obs=10.0, gamma=0.99)


def _build_vec_env(
    env_factory,
    *,
    n_envs: int,
    seed: int,
    training: bool,
    stats_path: Path | None = None,
) -> Any:
    n_envs = max(1, int(n_envs))

    vec_env = DummyVecEnv([env_factory(index) for index in range(n_envs)])
    vec_env.seed(seed)
    if stats_path is not None:
        return _load_vecnormalize_stats(stats_path, vec_env, training=training)
    return VecNormalize(vec_env, training=training, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99)


def _build_dino_vec_env(
    config: SimConfig | None,
    *,
    n_envs: int,
    seed: int,
    training: bool,
    stats_path: Path | None = None,
) -> Any:
    config = config or SimConfig(seed=seed)

    def make_env(rank: int):
        def _init():
            return DinoEnv(config=replace(config))

        return _init

    return _build_vec_env(make_env, n_envs=n_envs, seed=seed, training=training, stats_path=stats_path)


def _build_dino_dummy_vec_env(config: SimConfig | None, *, n_envs: int, seed: int) -> Any:
    config = config or SimConfig(seed=seed)

    def make_env(rank: int):
        def _init():
            return DinoEnv(config=replace(config))

        return _init

    vec_env = DummyVecEnv([make_env(index) for index in range(max(1, int(n_envs)))])
    vec_env.seed(seed)
    return vec_env


def _build_real_vec_env(
    *,
    real_config: RealGameConfig | None,
    vision_config: DinoVisionConfig | None,
    sim_config: SimConfig | None,
    env_config: RealEnvConfig | None,
    n_envs: int,
    seed: int,
    training: bool,
    stats_path: Path | None = None,
) -> Any:
    real_config = real_config or RealGameConfig(debug=False)
    vision_config = vision_config or DinoVisionConfig()
    sim_config = sim_config or SimConfig(seed=seed)
    env_config = env_config or RealEnvConfig(show_debug=real_config.debug, enable_action_gate=False)

    def make_env(rank: int):
        def _init():
            return DinoRealEnv(
                real_config=real_config,
                vision_config=vision_config,
                sim_config=sim_config,
                env_config=env_config,
            )

        return _init

    return _build_vec_env(make_env, n_envs=max(1, int(n_envs)), seed=seed, training=training, stats_path=stats_path)


def _append_history(artifacts: PPOArtifacts, entry: dict[str, Any]) -> None:
    history = _read_history(artifacts.training_history_path)
    history.append(_json_ready(entry))
    _write_json(artifacts.training_history_path, history)


def _append_eval_history(artifacts: PPOArtifacts, entry: dict[str, Any]) -> None:
    history = _read_history(artifacts.evaluation_history_path)
    history.append(_json_ready(entry))
    _write_json(artifacts.evaluation_history_path, history)


def _timestamped_payload(stage: str) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{stage}-{uuid.uuid4().hex[:8]}"


def _make_run_dir(artifacts: PPOArtifacts, stage: str) -> Path:
    run_dir = artifacts.run_dir(stage)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _select_best_metrics(artifacts: PPOArtifacts, stage: str) -> dict[str, Any] | None:
    path = artifacts.best_real_eval_path if stage == "real" else artifacts.best_sim_eval_path
    raw = _read_json(path)
    return raw if isinstance(raw, dict) else None


def _update_best_metrics(
    artifacts: PPOArtifacts,
    *,
    stage: str,
    metrics: dict[str, Any],
    model_path: Path,
    vecnormalize_path: Path | None = None,
) -> bool:
    def is_better(candidate: dict[str, Any], reference: dict[str, Any] | None) -> bool:
        if reference is None:
            return True
        best_reward = float(reference.get("mean_reward", float("-inf")))
        new_reward = float(candidate.get("mean_reward", float("-inf")))
        best_passed = float(reference.get("mean_passed_obstacles", float("-inf")))
        new_passed = float(candidate.get("mean_passed_obstacles", float("-inf")))
        if new_reward > best_reward + 1e-9:
            return True
        if abs(new_reward - best_reward) <= 1e-9 and new_passed > best_passed:
            return True
        return False

    stage_metrics = _select_best_metrics(artifacts, stage)
    stage_improved = is_better(metrics, stage_metrics)
    if stage_improved:
        best_metrics_path = artifacts.best_real_eval_path if stage == "real" else artifacts.best_sim_eval_path
        best_model_path = artifacts.best_real_model_path if stage == "real" else artifacts.best_sim_model_path
        best_vecnormalize_path = artifacts.best_real_vecnormalize_path if stage == "real" else artifacts.best_sim_vecnormalize_path
        _write_json(best_metrics_path, metrics)
        _copy_file(model_path, best_model_path)
        if vecnormalize_path is None:
            vecnormalize_path = _vecnormalize_companion_path(model_path)
        if vecnormalize_path.exists():
            _copy_file(vecnormalize_path, best_vecnormalize_path)

    global_metrics = _read_json(artifacts.best_metrics_path)
    if is_dataclass(global_metrics):  # pragma: no cover - defensive; _read_json never returns dataclass
        global_metrics = asdict(global_metrics)
    global_reference = global_metrics if isinstance(global_metrics, dict) else stage_metrics
    if is_better(metrics, global_reference):
        global_payload = {"stage": stage, **metrics, "model_path": str(artifacts.best_model_path)}
        _copy_file(model_path, artifacts.best_model_path)
        if vecnormalize_path is None:
            vecnormalize_path = _vecnormalize_companion_path(model_path)
        if vecnormalize_path.exists():
            _copy_file(vecnormalize_path, artifacts.best_model_vecnormalize_path)
            global_payload["vecnormalize_path"] = str(artifacts.best_model_vecnormalize_path)
        _write_json(artifacts.best_metrics_path, global_payload)
        return True

    return stage_improved


def _resolve_best_source_model(artifacts: PPOArtifacts, source_model_path: str | Path | None = None) -> Path:
    if source_model_path is not None:
        path = Path(source_model_path)
        if not path.exists():
            raise FileNotFoundError(f"Sim PPO checkpoint not found: {path}")
        return path
    if artifacts.best_sim_model_path.exists():
        return artifacts.best_sim_model_path
    if artifacts.sim_model_path.exists():
        return artifacts.sim_model_path
    raise FileNotFoundError(
        f"No PPO sim model found in {artifacts.output_dir}. Run `python -m dinoia train-sim-ppo` first."
    )


def _select_model_path(artifacts: PPOArtifacts, model_path: str | Path | None = None) -> Path:
    if model_path is not None:
        path = Path(model_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"PPO model not found: {path}")

    for candidate in (
        artifacts.best_real_model_path,
        artifacts.best_sim_model_path,
        artifacts.real_model_path,
        artifacts.sim_model_path,
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No PPO model found in {artifacts.output_dir}. Run `python -m dinoia train-sim-ppo` first."
    )


def _artifacts(output_dir: str | Path | None = None) -> PPOArtifacts:
    return PPOArtifacts(output_dir=Path(output_dir) if output_dir is not None else DEFAULT_PPO_ARTIFACTS_DIR)


def _ensure_artifacts(output_dir: str | Path | None = None) -> PPOArtifacts:
    artifacts = _artifacts(output_dir)
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)
    return artifacts


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(data), indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_history(path: Path) -> list[dict[str, Any]]:
    raw = _read_json(path)
    return raw if isinstance(raw, list) else []


def _write_latest_summary(artifacts: PPOArtifacts, entry: dict[str, Any]) -> None:
    lines = [
        "DinoIA PPO summary",
        f"stage={entry.get('stage')}",
        f"model={entry.get('model_path')}",
        f"best_model={entry.get('best_model_path') or 'none'}",
        f"source_model={entry.get('source_model_path') or 'none'}",
        f"total_timesteps={entry.get('total_timesteps')}",
        f"best_eval_mean_reward={entry.get('best_eval_mean_reward') if entry.get('best_eval_mean_reward') is not None else 'none'}",
        f"best_eval_mean_passed_obstacles={entry.get('best_eval_mean_passed_obstacles') if entry.get('best_eval_mean_passed_obstacles') is not None else 'none'}",
        f"created_at={entry.get('created_at')}",
    ]
    artifacts.latest_summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dataclass_from_mapping(cls, mapping: dict[str, Any]):
    names = {field.name for field in fields(cls)}
    values = {key: value for key, value in mapping.items() if key in names}
    return cls(**values)


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _action_to_int(action: Any) -> int:
    return int(np.asarray(action).item())


def _action_name(action: int) -> str:
    if action == 1:
        return "jump"
    if action == 2:
        return "duck"
    return "noop"


@dataclass(slots=True)
class CurriculumPhase:
    name: str
    timesteps: int
    config: SimConfig


def _build_sim_curriculum(base_config: SimConfig, total_timesteps: int) -> list[CurriculumPhase]:
    weights = [0.2, 0.3, 0.5]
    counts = [max(0, int(total_timesteps * weight)) for weight in weights]
    while sum(counts) < total_timesteps:
        counts[-1] += 1
    while sum(counts) > total_timesteps:
        for index in range(len(counts) - 1):
            if counts[index] > 0:
                counts[index] -= 1
                break
        else:
            counts[-1] -= 1

    easy = replace(
        base_config,
        domain_randomization=False,
        domain_randomization_strength=0.0,
        observation_noise=0.0,
        action_latency_steps=0,
        observation_latency_steps=0,
        teacher_match_reward=min(base_config.teacher_match_reward, 0.06),
        teacher_mismatch_penalty=max(base_config.teacher_mismatch_penalty, -0.08),
        unnecessary_jump_penalty=min(base_config.unnecessary_jump_penalty, -0.05),
        unnecessary_duck_penalty=min(base_config.unnecessary_duck_penalty, -0.06),
        missed_action_penalty=min(base_config.missed_action_penalty, -0.12),
        enable_teacher_shaping=True,
    )
    medium = replace(
        base_config,
        domain_randomization=True,
        domain_randomization_strength=max(0.18, base_config.domain_randomization_strength * 1.0),
        observation_noise=max(base_config.observation_noise, 0.025),
        action_latency_steps=max(1, base_config.action_latency_steps),
        observation_latency_steps=max(1, base_config.observation_latency_steps),
        teacher_match_reward=min(base_config.teacher_match_reward, 0.03),
        teacher_mismatch_penalty=max(base_config.teacher_mismatch_penalty, -0.08),
        unnecessary_jump_penalty=min(base_config.unnecessary_jump_penalty, -0.06),
        unnecessary_duck_penalty=min(base_config.unnecessary_duck_penalty, -0.08),
        missed_action_penalty=min(base_config.missed_action_penalty, -0.15),
        enable_teacher_shaping=True,
    )
    hard = replace(
        base_config,
        domain_randomization=True,
        domain_randomization_strength=max(0.24, base_config.domain_randomization_strength * 1.2),
        observation_noise=max(base_config.observation_noise, 0.035),
        action_latency_steps=max(2, base_config.action_latency_steps + 1),
        observation_latency_steps=max(2, base_config.observation_latency_steps + 1),
        teacher_match_reward=0.0,
        teacher_mismatch_penalty=0.0,
        unnecessary_jump_penalty=min(base_config.unnecessary_jump_penalty, -0.08),
        unnecessary_duck_penalty=min(base_config.unnecessary_duck_penalty, -0.1),
        missed_action_penalty=min(base_config.missed_action_penalty, -0.18),
        enable_teacher_shaping=False,
    )
    phases = [
        CurriculumPhase("easy", counts[0], easy),
        CurriculumPhase("medium", counts[1], medium),
        CurriculumPhase("hard", counts[2], hard),
    ]
    return [phase for phase in phases if phase.timesteps > 0]


def _build_eval_config(base_config: SimConfig | None = None) -> SimConfig:
    config = base_config or SimConfig()
    return replace(
        config,
        domain_randomization=False,
        domain_randomization_strength=0.0,
        observation_noise=0.0,
        action_latency_steps=0,
        observation_latency_steps=0,
        enable_teacher_shaping=False,
    )


def _build_transfer_eval_config(base_config: SimConfig | None = None) -> SimConfig:
    config = base_config or SimConfig()
    return replace(
        config,
        domain_randomization=True,
        domain_randomization_strength=max(0.22, config.domain_randomization_strength),
        observation_noise=max(config.observation_noise, 0.025),
        action_latency_steps=max(1, config.action_latency_steps),
        observation_latency_steps=max(1, config.observation_latency_steps),
        enable_teacher_shaping=False,
    )


def _build_teacher_state(
    *,
    config: SimConfig,
    floor_y: float,
    player_y: float,
    player_vy: float,
    grounded: bool,
    ducking: bool,
    current_speed: float,
    nearest: Any | None,
    second: Any | None,
    obstacle_count: int,
) -> VisionState:
    frame = np.zeros((config.screen_height, config.screen_width, 3), dtype=np.uint8)
    playfield = frame.copy()
    player_box = Rect(config.player_x, int(player_y), config.player_width, config.duck_height if ducking and grounded else config.stand_height)

    def _to_detection(obstacle: SimObstacle | None) -> ObstacleDetection | None:
        if obstacle is None:
            return None
        rect = Rect(int(obstacle.left), int(obstacle.top), int(obstacle.w), int(obstacle.h))
        if rect.right < player_box.left - 40:
            return None
        return ObstacleDetection(
            rect=rect,
            label=obstacle.kind,
            distance_px=max(0.0, float(rect.x - (player_box.x + player_box.w))),
            speed_px_s=0.0,
        )

    detections = [_to_detection(nearest), _to_detection(second)]
    detections = [item for item in detections if item is not None]
    detections.sort(key=lambda item: item.distance_px)
    nearest_detection = detections[0] if detections else None
    return VisionState(
        timestamp=0.0,
        frame=frame,
        playfield=playfield,
        player_box=player_box,
        ground_y=int(floor_y),
        obstacles=detections,
        nearest_obstacle=nearest_detection,
        nearest_distance_px=nearest_detection.distance_px if nearest_detection is not None else None,
        estimated_speed_px_s=current_speed,
        estimated_player_vy_px_s=player_vy,
        game_over=False,
    )


def select_best_ppo_model(output_dir: str | Path | None = None) -> Path | None:
    artifacts = _artifacts(output_dir)
    best_metrics = _read_json(artifacts.best_metrics_path)
    if isinstance(best_metrics, dict):
        model_path = best_metrics.get("model_path")
        if model_path:
            path = Path(model_path)
            if path.exists():
                return path
    for candidate in (
        artifacts.best_real_model_path,
        artifacts.best_sim_model_path,
        artifacts.real_model_path,
        artifacts.sim_model_path,
    ):
        if candidate.exists():
            return candidate
    return None


def load_sim_config_from_artifacts(output_dir: str | Path | None = None) -> SimConfig | None:
    artifacts = _artifacts(output_dir)
    raw = _read_json(artifacts.sim_training_config_path)
    if not isinstance(raw, dict):
        return None
    sim_config = raw.get("sim_config")
    if not isinstance(sim_config, dict):
        return None
    return _dataclass_from_mapping(SimConfig, sim_config)


def evaluate_sim_ppo(
    *,
    model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    episodes: int = 10,
    sim_config: SimConfig | None = None,
    seed: int = 42,
    device: str = "cpu",
    deterministic: bool = True,
) -> PPOEvalResult:
    episodes = _positive_int("episodes", episodes)
    artifacts = _ensure_artifacts(output_dir)
    selected_model_path = _select_model_path(artifacts, model_path)
    base_config = sim_config or load_sim_config_from_artifacts(artifacts.output_dir) or SimConfig(seed=seed)
    eval_config = _build_eval_config(base_config)
    PPO = _load_ppo()
    stats_path = _vecnormalize_companion_path(selected_model_path)
    env = _build_dino_vec_env(
        eval_config,
        n_envs=1,
        seed=seed,
        training=False,
        stats_path=stats_path if stats_path.exists() else None,
    )
    model = PPO.load(str(selected_model_path), env=env, device=device)

    action_counts = Counter({"noop": 0, "jump": 0, "duck": 0})
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    passed_values: list[int] = []
    total_steps = 0

    try:
        for episode_index in range(episodes):
            env.seed(seed + episode_index)
            obs = env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            episode_passed = 0

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                action_int = _action_to_int(action)
                action_counts[_action_name(action_int)] += 1
                obs, reward, dones, infos = env.step(np.array([action_int], dtype=np.int64))
                reward_value = float(np.asarray(reward).reshape(-1)[0])
                info = infos[0]
                done = bool(np.asarray(dones).reshape(-1)[0])
                episode_reward += reward_value
                episode_length += 1
                total_steps += 1
                episode_passed = max(episode_passed, int(info.get("passed_obstacles", 0)))

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            passed_values.append(episode_passed)
    finally:
        env.close()

    metrics = {
        "kind": "sim",
        "model_path": str(selected_model_path),
        "episodes": episodes,
        "steps": total_steps,
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "reward_std": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        "mean_passed_obstacles": float(np.mean(passed_values)) if passed_values else 0.0,
        "max_passed_obstacles": int(max(passed_values)) if passed_values else 0,
        "action_counts": dict(action_counts),
        "action_distribution": {
            action: (count / max(1, sum(action_counts.values()))) for action, count in action_counts.items()
        },
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "created_at": _utc_now(),
    }
    run_dir = _make_run_dir(artifacts, "eval")
    _write_json(run_dir / "eval.json", metrics)
    _write_json(artifacts.latest_eval_path, metrics)
    _append_eval_history(artifacts, metrics)
    return PPOEvalResult(
        model_path=selected_model_path,
        run_dir=run_dir,
        episodes=episodes,
        steps=total_steps,
        mean_reward=metrics["mean_reward"],
        reward_std=metrics["reward_std"],
        mean_passed_obstacles=metrics["mean_passed_obstacles"],
        max_passed_obstacles=metrics["max_passed_obstacles"],
        action_counts=metrics["action_counts"],
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
    )


def train_sim_ppo(
    *,
    total_timesteps: int = 250_000,
    output_dir: str | Path | None = None,
    source_model_path: str | Path | None = None,
    sim_config: SimConfig | None = None,
    seed: int = 42,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    n_steps: int = 1024,
    batch_size: int = 64,
    gamma: float = 0.99,
    entropy_coef: float = 0.02,
    clip_range: float = 0.15,
    gae_lambda: float = 0.95,
    n_epochs: int = 10,
    target_kl: float = 0.04,
    device: str = "cpu",
    verbose: int = 1,
) -> PPOTrainResult:
    total_timesteps = _positive_int("total_timesteps", total_timesteps)
    n_envs = _positive_int("n_envs", n_envs)
    artifacts = _ensure_artifacts(output_dir)
    base_config = sim_config or SimConfig(seed=seed)
    run_dir = _make_run_dir(artifacts, "sim")
    phases = _build_sim_curriculum(base_config, total_timesteps)
    PPO = _load_ppo()
    source_path = None
    if source_model_path is not None:
        source_path = Path(source_model_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Sim PPO checkpoint not found: {source_path}")
    source_stats_path = _vecnormalize_companion_path(source_path) if source_path is not None else None
    hyperparameters = {
        "learning_rate": learning_rate,
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "gamma": gamma,
        "ent_coef": entropy_coef,
        "clip_range": clip_range,
        "gae_lambda": gae_lambda,
        "n_epochs": int(n_epochs),
        "target_kl": target_kl,
        "seed": int(seed),
        "device": device,
    }
    config_doc = {
        "stage": "sim",
        "created_at": _utc_now(),
        "policy": "MlpPolicy",
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "source_model_path": source_path,
        "source_vecnormalize_path": source_stats_path if source_stats_path and source_stats_path.exists() else None,
        "curriculum": [
            {"name": phase.name, "timesteps": phase.timesteps, "sim_config": phase.config} for phase in phases
        ],
        "hyperparameters": hyperparameters,
        "sim_config": base_config,
        "model_path": artifacts.sim_model_path,
        "vecnormalize_path": artifacts.sim_vecnormalize_path,
        "run_dir": run_dir,
    }

    cumulative_timesteps = 0
    best_phase_eval: PPOEvalResult | None = None
    best_phase_model_path: Path | None = None
    phase_records: list[dict[str, Any]] = []
    transfer_eval_config = _build_transfer_eval_config(base_config)
    model = None
    try:
        current_stats_path = source_stats_path if source_stats_path and source_stats_path.exists() else None
        for index, phase in enumerate(phases):
            train_env = _build_dino_vec_env(
                phase.config,
                n_envs=n_envs,
                seed=seed + index,
                training=True,
                stats_path=current_stats_path,
            )
            try:
                if model is None:
                    if source_path is not None:
                        model = PPO.load(str(source_path), env=train_env, device=device)
                    else:
                        model = PPO("MlpPolicy", train_env, verbose=verbose, **hyperparameters)
                else:
                    model.set_env(train_env)

                reset_timesteps = index == 0 and source_path is None
                model.learn(total_timesteps=phase.timesteps, reset_num_timesteps=reset_timesteps)

                cumulative_timesteps += phase.timesteps
                phase_model_path = run_dir / f"{index + 1:02d}-{phase.name}.zip"
                model.save(str(phase_model_path))
                phase_stats_path = _vecnormalize_companion_path(phase_model_path)
                _save_vecnormalize(train_env, phase_stats_path)
                phase_eval = evaluate_sim_ppo(
                    model_path=phase_model_path,
                    output_dir=artifacts.output_dir,
                    episodes=5,
                    sim_config=base_config,
                    seed=seed + index,
                    device=device,
                )
                phase_transfer_eval = evaluate_sim_ppo(
                    model_path=phase_model_path,
                    output_dir=artifacts.output_dir,
                    episodes=3,
                    sim_config=transfer_eval_config,
                    seed=seed + 1000 + index,
                    device=device,
                )
                phase_records.append(
                    {
                        "name": phase.name,
                        "timesteps": phase.timesteps,
                        "model_path": phase_model_path,
                        "vecnormalize_path": phase_stats_path,
                        "evaluation_path": phase_eval.run_dir / "eval.json",
                        "mean_reward": phase_eval.mean_reward,
                        "mean_passed_obstacles": phase_eval.mean_passed_obstacles,
                        "transfer_mean_reward": phase_transfer_eval.mean_reward,
                        "transfer_mean_passed_obstacles": phase_transfer_eval.mean_passed_obstacles,
                    }
                )
                if best_phase_eval is None or (
                    phase_transfer_eval.mean_reward > best_phase_eval.mean_reward + 1e-9
                    or (
                        abs(phase_transfer_eval.mean_reward - best_phase_eval.mean_reward) <= 1e-9
                        and phase_transfer_eval.mean_passed_obstacles > best_phase_eval.mean_passed_obstacles
                    )
                ):
                    best_phase_eval = phase_transfer_eval
                    best_phase_model_path = phase_model_path
                current_stats_path = phase_stats_path
            finally:
                train_env.close()
    finally:
        if model is not None:
            model.save(str(artifacts.sim_model_path))
        if "current_stats_path" in locals() and current_stats_path is not None and current_stats_path.exists():
            _copy_file(current_stats_path, artifacts.sim_vecnormalize_path)

    _write_json(artifacts.sim_training_config_path, {**config_doc, "phases": phase_records})
    entry = {
        "stage": "sim",
        "created_at": config_doc["created_at"],
        "model_path": artifacts.sim_model_path,
        "vecnormalize_path": artifacts.sim_vecnormalize_path,
        "source_model_path": source_path,
        "source_vecnormalize_path": source_stats_path if source_stats_path and source_stats_path.exists() else None,
        "config_path": artifacts.sim_training_config_path,
        "total_timesteps": cumulative_timesteps,
        "run_dir": run_dir,
        "best_eval_mean_reward": best_phase_eval.mean_reward if best_phase_eval is not None else None,
        "best_eval_mean_passed_obstacles": best_phase_eval.mean_passed_obstacles if best_phase_eval is not None else None,
    }

    latest_eval = evaluate_sim_ppo(
        model_path=artifacts.sim_model_path,
        output_dir=artifacts.output_dir,
        episodes=10,
        sim_config=base_config,
        seed=seed,
        device=device,
    )
    latest_transfer_eval = evaluate_sim_ppo(
        model_path=artifacts.sim_model_path,
        output_dir=artifacts.output_dir,
        episodes=10,
        sim_config=transfer_eval_config,
        seed=seed + 10000,
        device=device,
    )
    candidate_eval = best_phase_eval or latest_transfer_eval
    candidate_model_path = best_phase_model_path or artifacts.sim_model_path
    candidate_stats_path = _vecnormalize_companion_path(candidate_model_path)
    _update_best_metrics(
        artifacts,
        stage="sim",
        metrics={
            "kind": "sim",
            "model_path": str(candidate_model_path),
            "vecnormalize_path": str(candidate_stats_path) if candidate_stats_path.exists() else None,
            "mean_reward": candidate_eval.mean_reward,
            "mean_passed_obstacles": candidate_eval.mean_passed_obstacles,
            "reward_std": candidate_eval.reward_std,
            "episodes": candidate_eval.episodes,
            "steps": candidate_eval.steps,
            "action_counts": candidate_eval.action_counts,
            "transfer_eval_mean_reward": latest_transfer_eval.mean_reward,
            "transfer_eval_mean_passed_obstacles": latest_transfer_eval.mean_passed_obstacles,
            "created_at": _utc_now(),
        },
        model_path=candidate_model_path,
        vecnormalize_path=candidate_stats_path if candidate_stats_path.exists() else None,
    )
    entry["best_model_path"] = select_best_ppo_model(artifacts.output_dir)
    _append_history(artifacts, entry)
    _write_latest_summary(artifacts, entry)

    return PPOTrainResult(
        stage="sim",
        model_path=artifacts.sim_model_path,
        best_model_path=artifacts.best_sim_model_path if artifacts.best_sim_model_path.exists() else None,
        run_dir=run_dir,
        total_timesteps=total_timesteps,
        config_path=artifacts.sim_training_config_path,
        history_path=artifacts.training_history_path,
        evaluation_path=artifacts.latest_eval_path,
        source_model_path=source_path,
    )


def train_real_ppo(
    *,
    total_timesteps: int = 5_000,
    output_dir: str | Path | None = None,
    source_model_path: str | Path | None = None,
    real_config: RealGameConfig | None = None,
    vision_config: DinoVisionConfig | None = None,
    sim_config: SimConfig | None = None,
    env_config: RealEnvConfig | None = None,
    seed: int = 42,
    device: str = "cpu",
    learning_rate: float = 1e-5,
) -> PPOTrainResult:
    total_timesteps = _positive_int("total_timesteps", total_timesteps)
    artifacts = _ensure_artifacts(output_dir)
    run_dir = _make_run_dir(artifacts, "real")
    source_path = _resolve_best_source_model(artifacts, source_model_path)
    source_stats_path = _vecnormalize_companion_path(source_path)

    real_config = real_config or RealGameConfig(debug=False)
    sim_config = sim_config or load_sim_config_from_artifacts(artifacts.output_dir) or SimConfig(seed=seed)
    env_config = env_config or RealEnvConfig(show_debug=real_config.debug, enable_action_gate=False)
    PPO = _load_ppo()
    env = _build_real_vec_env(
        real_config=real_config,
        vision_config=vision_config,
        sim_config=sim_config,
        env_config=env_config,
        n_envs=1,
        seed=seed,
        training=True,
        stats_path=source_stats_path if source_stats_path.exists() else None,
    )

    config_doc = {
        "stage": "real",
        "created_at": _utc_now(),
        "policy": "MlpPolicy",
        "total_timesteps": total_timesteps,
        "source_model_path": source_path,
        "model_path": artifacts.real_model_path,
        "vecnormalize_path": artifacts.real_vecnormalize_path,
        "source_vecnormalize_path": source_stats_path if source_stats_path.exists() else None,
        "real_config": real_config,
        "vision_config": vision_config or DinoVisionConfig(),
        "sim_config": sim_config,
        "env_config": env_config,
        "hyperparameters": {
            "seed": int(seed),
            "device": device,
            "learning_rate": learning_rate,
            "reset_num_timesteps": False,
        },
        "run_dir": run_dir,
    }

    try:
        model = PPO.load(str(source_path), env=env, device=device, custom_objects={"learning_rate": learning_rate})
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        model.save(str(artifacts.real_model_path))
        _save_vecnormalize(env, artifacts.real_vecnormalize_path)
    finally:
        env.close()

    _write_json(run_dir / "real_training_config.json", config_doc)
    _write_json(artifacts.real_training_config_path, config_doc)
    entry = {
        "stage": "real",
        "created_at": config_doc["created_at"],
        "model_path": artifacts.real_model_path,
        "source_model_path": source_path,
        "config_path": artifacts.real_training_config_path,
        "total_timesteps": total_timesteps,
        "run_dir": run_dir,
    }

    eval_result = evaluate_sim_ppo(
        model_path=artifacts.real_model_path,
        output_dir=artifacts.output_dir,
        episodes=10,
        sim_config=sim_config,
        seed=seed,
        device=device,
    )
    _update_best_metrics(
        artifacts,
        stage="real",
        metrics={
            "kind": "real",
            "model_path": str(artifacts.real_model_path),
            "vecnormalize_path": str(artifacts.real_vecnormalize_path),
            "source_model_path": str(source_path),
            "mean_reward": eval_result.mean_reward,
            "mean_passed_obstacles": eval_result.mean_passed_obstacles,
            "reward_std": eval_result.reward_std,
            "episodes": eval_result.episodes,
            "steps": eval_result.steps,
            "action_counts": eval_result.action_counts,
            "created_at": _utc_now(),
        },
        model_path=artifacts.real_model_path,
        vecnormalize_path=artifacts.real_vecnormalize_path,
    )
    entry["best_model_path"] = select_best_ppo_model(artifacts.output_dir)
    entry["best_eval_mean_reward"] = eval_result.mean_reward
    entry["best_eval_mean_passed_obstacles"] = eval_result.mean_passed_obstacles
    _append_history(artifacts, entry)
    _write_latest_summary(artifacts, entry)
    return PPOTrainResult(
        stage="real",
        model_path=artifacts.real_model_path,
        best_model_path=artifacts.best_real_model_path if artifacts.best_real_model_path.exists() else None,
        run_dir=run_dir,
        source_model_path=source_path,
        total_timesteps=total_timesteps,
        config_path=artifacts.real_training_config_path,
        history_path=artifacts.training_history_path,
        evaluation_path=artifacts.latest_eval_path,
    )


def play_sim_ppo(
    *,
    model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    duration_s: float = 30.0,
    sim_config: SimConfig | None = None,
    display: bool = True,
    device: str = "cpu",
    deterministic: bool = True,
    seed: int = 42,
) -> PPORunResult:
    artifacts = _artifacts(output_dir)
    selected_model_path = _select_model_path(artifacts, model_path)
    sim_config = sim_config or load_sim_config_from_artifacts(artifacts.output_dir) or SimConfig(seed=seed)
    PPO = _load_ppo()
    stats_path = _vecnormalize_companion_path(selected_model_path)
    env = _build_dino_vec_env(
        sim_config,
        n_envs=1,
        seed=seed,
        training=False,
        stats_path=stats_path if stats_path.exists() else None,
    )
    model = PPO.load(str(selected_model_path), env=env, device=device)

    steps = 0
    episodes = 1
    total_reward = 0.0
    passed_obstacles = 0
    noops = jumps = ducks = 0
    action_counts = Counter({"noop": 0, "jump": 0, "duck": 0})
    deadline = time.time() + max(0.0, float(duration_s))

    try:
        env.seed(seed)
        obs = env.reset()
        while time.time() < deadline:
            action, _ = model.predict(obs, deterministic=deterministic)
            action_int = _action_to_int(action)
            action_counts[_action_name(action_int)] += 1
            if action_int == 1:
                jumps += 1
            elif action_int == 2:
                ducks += 1
            else:
                noops += 1

            obs, reward, dones, infos = env.step(np.array([action_int], dtype=np.int64))
            total_reward += float(np.asarray(reward).reshape(-1)[0])
            passed_obstacles = max(passed_obstacles, int(infos[0].get("passed_obstacles", 0)))
            steps += 1
            if display:
                env.render()
            if bool(np.asarray(dones).reshape(-1)[0]):
                episodes += 1
                env.seed(seed + episodes)
                obs = env.reset()
    finally:
        env.close()

    return PPORunResult(
        mode="sim",
        model_path=selected_model_path,
        steps=steps,
        reward=total_reward,
        passed_obstacles=passed_obstacles,
        episodes=episodes,
        action_counts=dict(action_counts),
        noops=noops,
        jumps=jumps,
        ducks=ducks,
    )


def play_real_ppo(
    *,
    model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    duration_s: float | None = None,
    real_config: RealGameConfig | None = None,
    vision_config: DinoVisionConfig | None = None,
    sim_config: SimConfig | None = None,
    env_config: RealEnvConfig | None = None,
    device: str = "cpu",
    deterministic: bool = True,
) -> PPORunResult:
    artifacts = _artifacts(output_dir)
    selected_model_path = _select_model_path(artifacts, model_path)
    real_config = real_config or RealGameConfig(debug=True)
    sim_config = sim_config or load_sim_config_from_artifacts(artifacts.output_dir) or SimConfig()
    env_config = env_config or RealEnvConfig(show_debug=real_config.debug, enable_action_gate=False)
    PPO = _load_ppo()
    stats_path = _vecnormalize_companion_path(selected_model_path)
    env = _build_real_vec_env(
        real_config=real_config,
        vision_config=vision_config,
        sim_config=sim_config,
        env_config=env_config,
        n_envs=1,
        seed=42,
        training=False,
        stats_path=stats_path if stats_path.exists() else None,
    )
    model = PPO.load(str(selected_model_path), env=env, device=device)

    steps = 0
    episodes = 1
    total_reward = 0.0
    passed_obstacles = 0
    noops = jumps = ducks = 0
    action_counts = Counter({"noop": 0, "jump": 0, "duck": 0})
    start = time.time()

    try:
        env.seed(42)
        obs = env.reset()
        while duration_s is None or time.time() - start < float(duration_s):
            action, _ = model.predict(obs, deterministic=deterministic)
            action_int = _action_to_int(action)
            action_counts[_action_name(action_int)] += 1
            if action_int == 1:
                jumps += 1
            elif action_int == 2:
                ducks += 1
            else:
                noops += 1

            obs, reward, dones, infos = env.step(np.array([action_int], dtype=np.int64))
            total_reward += float(np.asarray(reward).reshape(-1)[0])
            passed_obstacles = max(passed_obstacles, int(infos[0].get("passed_obstacles", 0)))
            steps += 1

            if real_config.debug:
                import cv2

                frame = env.render()
                if frame is not None:
                    cv2.imshow("DinoIA PPO Real", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if bool(np.asarray(dones).reshape(-1)[0]):
                episodes += 1
                env.seed(42 + episodes)
                obs = env.reset()
    finally:
        env.close()
        if real_config.debug:
            import cv2

            cv2.destroyAllWindows()

    return PPORunResult(
        mode="real",
        model_path=selected_model_path,
        steps=steps,
        reward=total_reward,
        passed_obstacles=passed_obstacles,
        episodes=episodes,
        action_counts=dict(action_counts),
        noops=noops,
        jumps=jumps,
        ducks=ducks,
    )


def summarize_ppo(output_dir: str | Path | None = None) -> PPOProjectResults:
    artifacts = _artifacts(output_dir)
    history = _read_history(artifacts.training_history_path)
    evaluation_history = _read_history(artifacts.evaluation_history_path)
    return PPOProjectResults(
        output_dir=artifacts.output_dir,
        sim_model_path=artifacts.sim_model_path if artifacts.sim_model_path.exists() else None,
        best_sim_model_path=artifacts.best_sim_model_path if artifacts.best_sim_model_path.exists() else None,
        real_model_path=artifacts.real_model_path if artifacts.real_model_path.exists() else None,
        best_real_model_path=artifacts.best_real_model_path if artifacts.best_real_model_path.exists() else None,
        sim_vecnormalize_path=artifacts.sim_vecnormalize_path if artifacts.sim_vecnormalize_path.exists() else None,
        best_sim_vecnormalize_path=artifacts.best_sim_vecnormalize_path
        if artifacts.best_sim_vecnormalize_path.exists()
        else None,
        real_vecnormalize_path=artifacts.real_vecnormalize_path if artifacts.real_vecnormalize_path.exists() else None,
        best_real_vecnormalize_path=artifacts.best_real_vecnormalize_path
        if artifacts.best_real_vecnormalize_path.exists()
        else None,
        best_model_vecnormalize_path=artifacts.best_model_vecnormalize_path
        if artifacts.best_model_vecnormalize_path.exists()
        else None,
        sim_training_config_path=artifacts.sim_training_config_path
        if artifacts.sim_training_config_path.exists()
        else None,
        real_training_config_path=artifacts.real_training_config_path
        if artifacts.real_training_config_path.exists()
        else None,
        history_path=artifacts.training_history_path if artifacts.training_history_path.exists() else None,
        evaluation_history_path=artifacts.evaluation_history_path if artifacts.evaluation_history_path.exists() else None,
        latest_eval_path=artifacts.latest_eval_path if artifacts.latest_eval_path.exists() else None,
        best_sim_eval_path=artifacts.best_sim_eval_path if artifacts.best_sim_eval_path.exists() else None,
        best_real_eval_path=artifacts.best_real_eval_path if artifacts.best_real_eval_path.exists() else None,
        best_metrics_path=artifacts.best_metrics_path if artifacts.best_metrics_path.exists() else None,
        summary_path=artifacts.latest_summary_path if artifacts.latest_summary_path.exists() else None,
        history=history,
        evaluation_history=evaluation_history,
    )
