from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Sequence

import gymnasium as gym
import cv2
import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

from .config import SimConfig
from .observations import OBSERVATION_SIZE
from .sim.env import DinoSimEnv

GYM_PARALLEL_ARTIFACTS_DIR = Path("artifacts") / "gym_parallel"


@dataclass(slots=True)
class GymParallelConfig:
    generations: int = 20
    population: int = 16
    workers: int = 4
    rollout_steps: int = 720
    mutation_scale: float = 0.18
    elite_fraction: float = 0.25
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: GYM_PARALLEL_ARTIFACTS_DIR)
    sim_config: SimConfig = field(default_factory=SimConfig)


@dataclass(slots=True)
class GymParallelResult:
    generation: int
    best_reward: float
    mean_reward: float
    best_passed_obstacles: int
    best_steps: int
    best_policy_path: str
    checkpoint_path: str


@dataclass(slots=True)
class GymParallelRunResult:
    model_path: str
    frames: int
    reward: float
    passed_obstacles: int
    episodes: int


@dataclass(slots=True)
class GymParallelHandoff:
    source_policy_path: str
    good_enough: bool
    best_reward: float
    best_passed_obstacles: int
    best_steps: int
    recommended_sim_config: dict[str, Any]
    recommended_neat_config: dict[str, Any]


class GymParallelArtifacts:
    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir) if output_dir is not None else GYM_PARALLEL_ARTIFACTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "training_history.json"
        self.summary_path = self.output_dir / "latest_summary.txt"
        self.policy_path = self.output_dir / "best_policy.npz"
        self.config_path = self.output_dir / "gym_parallel_config.json"
        self.handoff_path = self.output_dir / "handoff.json"

    def append_generation(self, entry: dict[str, Any]) -> None:
        history = self.read_history()
        history.append(entry)
        self.history_path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
        self.summary_path.write_text(format_latest_summary(entry), encoding="utf-8")

    def read_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        raw = json.loads(self.history_path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else []

    def save_policy(self, policy: "LinearPolicy") -> None:
        np.savez_compressed(self.policy_path, weights=policy.weights, bias=policy.bias)

    def save_config(self, config: GymParallelConfig) -> None:
        payload = {
            "generations": config.generations,
            "population": config.population,
            "workers": config.workers,
            "rollout_steps": config.rollout_steps,
            "mutation_scale": config.mutation_scale,
            "elite_fraction": config.elite_fraction,
            "seed": config.seed,
            "sim_config": asdict(config.sim_config),
        }
        self.config_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def save_handoff(self, handoff: GymParallelHandoff) -> None:
        self.handoff_path.write_text(json.dumps(asdict(handoff), indent=2, default=str), encoding="utf-8")


def format_latest_summary(entry: dict[str, Any]) -> str:
    return "\n".join(
        [
            "DinoIA Gymnasium latest summary",
            f"generation={entry.get('generation')}",
            f"best_reward={entry.get('best_reward')}",
            f"mean_reward={entry.get('mean_reward')}",
            f"best_passed_obstacles={entry.get('best_passed_obstacles')}",
            f"best_steps={entry.get('best_steps')}",
            f"best_policy_path={entry.get('best_policy_path')}",
            f"checkpoint_path={entry.get('checkpoint_path')}",
        ]
    )


def build_handoff(entry: dict[str, Any], config: GymParallelConfig, policy_path: Path) -> GymParallelHandoff:
    best_reward = float(entry.get("best_reward", 0.0))
    best_passed = int(entry.get("best_passed_obstacles", 0))
    best_steps = int(entry.get("best_steps", 0))
    good_enough = best_reward >= 0.0 or best_passed > 0
    recommended_sim_config = asdict(config.sim_config)
    recommended_neat_config = {
        "generations": max(20, config.generations),
        "population": max(24, config.population * (2 if good_enough else 1)),
        "episode_distance_px": 2400.0 if good_enough else 1800.0,
        "seed": config.seed,
        "teacher_source": str(policy_path),
    }
    return GymParallelHandoff(
        source_policy_path=str(policy_path),
        good_enough=good_enough,
        best_reward=best_reward,
        best_passed_obstacles=best_passed,
        best_steps=best_steps,
        recommended_sim_config=recommended_sim_config,
        recommended_neat_config=recommended_neat_config,
    )


def load_gym_handoff(output_dir: str | Path | None = None) -> GymParallelHandoff | None:
    root = Path(output_dir) if output_dir is not None else GYM_PARALLEL_ARTIFACTS_DIR
    handoff_path = root / "handoff.json"
    if not handoff_path.exists():
        return None
    payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    return GymParallelHandoff(
        source_policy_path=str(payload.get("source_policy_path", "")),
        good_enough=bool(payload.get("good_enough", False)),
        best_reward=float(payload.get("best_reward", 0.0)),
        best_passed_obstacles=int(payload.get("best_passed_obstacles", 0)),
        best_steps=int(payload.get("best_steps", 0)),
        recommended_sim_config=dict(payload.get("recommended_sim_config", {})),
        recommended_neat_config=dict(payload.get("recommended_neat_config", {})),
    )


def annotate_parallel_frame(
    frame: np.ndarray,
    *,
    reward: float,
    passed_obstacles: int,
    action: str,
    episodes: int,
    frames: int,
) -> np.ndarray:
    annotated = frame.copy()
    lines = [
        f"action: {action}",
        f"reward: {reward:.2f}",
        f"passed: {passed_obstacles}",
        f"episodes: {episodes}",
        f"frames: {frames}",
    ]
    for index, text in enumerate(lines):
        y = 24 + index * 22
        cv2.putText(
            annotated,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return annotated


@dataclass(slots=True)
class LinearPolicy:
    weights: np.ndarray
    bias: np.ndarray

    @classmethod
    def random(cls, rng: np.random.Generator, obs_size: int = OBSERVATION_SIZE, action_size: int = 3) -> "LinearPolicy":
        weights = rng.normal(0.0, 0.45, size=(action_size, obs_size)).astype(np.float32)
        bias = rng.normal(0.0, 0.15, size=(action_size,)).astype(np.float32)
        return cls(weights=weights, bias=bias)

    def act_batch(self, observations: np.ndarray) -> np.ndarray:
        logits = observations @ self.weights.T + self.bias
        return np.asarray(np.argmax(logits, axis=1), dtype=np.int64)

    def act_one(self, observation: np.ndarray) -> int:
        logits = observation @ self.weights.T + self.bias
        return int(np.argmax(logits))

    def mutate(self, rng: np.random.Generator, scale: float) -> "LinearPolicy":
        weights = (self.weights + rng.normal(0.0, scale, size=self.weights.shape)).astype(np.float32)
        bias = (self.bias + rng.normal(0.0, scale * 0.5, size=self.bias.shape)).astype(np.float32)
        return LinearPolicy(weights=weights, bias=bias)

    @classmethod
    def load(cls, path: str | Path) -> "LinearPolicy":
        data = np.load(Path(path), allow_pickle=False)
        return cls(weights=data["weights"].astype(np.float32), bias=data["bias"].astype(np.float32))


def _make_sim_env(sim_config: SimConfig) -> DinoSimEnv:
    return DinoSimEnv(config=sim_config, render_mode=None)


class GymParallelTrainer:
    def __init__(
        self,
        *,
        config: GymParallelConfig | None = None,
        vector_env_cls: type | None = None,
    ) -> None:
        self.config = config or GymParallelConfig()
        self.vector_env_cls = vector_env_cls or AsyncVectorEnv
        self.artifacts = GymParallelArtifacts(self.config.output_dir)
        self.rng = np.random.default_rng(self.config.seed)
        self.artifacts.save_config(self.config)

    def train(self) -> Path:
        best_policy: LinearPolicy | None = None
        best_reward = float("-inf")
        best_passed = 0
        best_steps = 0

        population = self._make_initial_population()
        for generation in range(self.config.generations):
            results = self._evaluate_population(population, generation)
            results.sort(key=lambda item: item["reward"], reverse=True)
            top = results[0]

            if top["reward"] > best_reward or best_policy is None:
                best_reward = top["reward"]
                best_passed = int(top["passed_obstacles"])
                best_steps = int(top["steps"])
                best_policy = top["policy"]
                self.artifacts.save_policy(best_policy)

            entry = {
                "generation": generation,
                "best_reward": float(top["reward"]),
                "mean_reward": statistics.fmean(result["reward"] for result in results),
                "best_passed_obstacles": int(top["passed_obstacles"]),
                "best_steps": int(top["steps"]),
                "best_policy_path": str(self.artifacts.policy_path),
                "checkpoint_path": str(self.artifacts.output_dir / f"generation-{generation}.json"),
                "results": [
                    {
                        "reward": float(result["reward"]),
                        "passed_obstacles": int(result["passed_obstacles"]),
                        "steps": int(result["steps"]),
                    }
                    for result in results
                ],
            }
            (self.artifacts.output_dir / f"generation-{generation}.json").write_text(
                json.dumps(entry, indent=2, default=str),
                encoding="utf-8",
            )
            self.artifacts.append_generation(entry)
            print(
                "gym-train: generation={generation} best_reward={best:.3f} mean_reward={mean:.3f} "
                "passed={passed} steps={steps}".format(
                    generation=generation,
                    best=entry["best_reward"],
                    mean=entry["mean_reward"],
                    passed=entry["best_passed_obstacles"],
                    steps=entry["best_steps"],
                ),
                flush=True,
            )
            population = self._next_population(results, best_policy or top["policy"])

        if best_policy is None:
            raise RuntimeError("Gymnasium training did not produce a policy")
        self.artifacts.save_policy(best_policy)
        handoff = build_handoff(
            {
                "best_reward": best_reward,
                "best_passed_obstacles": best_passed,
                "best_steps": best_steps,
            },
            self.config,
            self.artifacts.policy_path,
        )
        self.artifacts.save_handoff(handoff)
        return self.artifacts.policy_path

    def _make_initial_population(self) -> list[LinearPolicy]:
        return [LinearPolicy.random(self.rng) for _ in range(self.config.population)]

    def _next_population(self, results: list[dict[str, Any]], elite: LinearPolicy) -> list[LinearPolicy]:
        elite_count = max(1, int(round(self.config.population * self.config.elite_fraction)))
        elites = [result["policy"] for result in results[:elite_count]]
        next_population: list[LinearPolicy] = [elite]
        while len(next_population) < self.config.population:
            parent = elites[len(next_population) % len(elites)]
            next_population.append(parent.mutate(self.rng, self.config.mutation_scale))
        return next_population[: self.config.population]

    def _evaluate_population(self, population: Sequence[LinearPolicy], generation: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        batch_size = max(1, int(self.config.workers))
        for batch_start in range(0, len(population), batch_size):
            batch = list(population[batch_start : batch_start + batch_size])
            rewards, passed, steps = self._evaluate_batch(batch, generation, batch_start)
            for index, policy in enumerate(batch):
                results.append(
                    {
                        "policy": policy,
                        "reward": float(rewards[index]),
                        "passed_obstacles": int(passed[index]),
                        "steps": int(steps[index]),
                    }
                )
        return results

    def _evaluate_batch(
        self,
        batch: Sequence[LinearPolicy],
        generation: int,
        batch_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        env_fns = [partial(_make_sim_env, self.config.sim_config) for _ in batch]
        vec_env = self.vector_env_cls(env_fns, autoreset_mode=AutoresetMode.NEXT_STEP)
        try:
            obs, _ = vec_env.reset(seed=self.config.seed + generation * 1000 + batch_index)
            total_reward = np.zeros(len(batch), dtype=np.float32)
            total_passed = np.zeros(len(batch), dtype=np.int32)
            steps = np.zeros(len(batch), dtype=np.int32)
            for _ in range(self.config.rollout_steps):
                actions = np.asarray([policy.act_batch(obs[index : index + 1])[0] for index, policy in enumerate(batch)], dtype=np.int64)
                obs, rewards, terminated, truncated, infos = vec_env.step(actions)
                total_reward += rewards.astype(np.float32)
                steps += 1
                passed = self._extract_vector_info(infos, "passed_obstacles")
                if passed is not None:
                    total_passed = np.maximum(total_passed, passed.astype(np.int32))
                if np.all(np.logical_or(terminated, truncated)):
                    break
            return total_reward, total_passed, steps
        finally:
            vec_env.close()

    @staticmethod
    def _extract_vector_info(infos: Any, key: str) -> np.ndarray | None:
        if not isinstance(infos, dict):
            return None
        value = infos.get(key)
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(value)


def train_parallel_gymnasium(
    *,
    config: GymParallelConfig | None = None,
    vector_env_cls: type | None = None,
) -> Path:
    trainer = GymParallelTrainer(config=config, vector_env_cls=vector_env_cls)
    return trainer.train()


def load_best_gym_policy(output_dir: str | Path | None = None) -> LinearPolicy | None:
    root = Path(output_dir) if output_dir is not None else GYM_PARALLEL_ARTIFACTS_DIR
    policy_path = root / "best_policy.npz"
    if not policy_path.exists():
        return None
    return LinearPolicy.load(policy_path)


def visualize_parallel_policy(
    *,
    model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    duration_s: float | None = 30.0,
    sim_config: SimConfig | None = None,
    env_factory: Callable[[], gym.Env] | None = None,
    display: bool = True,
) -> GymParallelRunResult:
    resolved_model_path: Path | None = Path(model_path) if model_path is not None else None
    if resolved_model_path is None:
        root = Path(output_dir) if output_dir is not None else GYM_PARALLEL_ARTIFACTS_DIR
        resolved_model_path = root / "best_policy.npz"
    if not resolved_model_path.exists():
        raise FileNotFoundError("No Gymnasium policy was found. Run train-sim-parallel first.")

    policy = LinearPolicy.load(resolved_model_path)
    if policy is None:
        raise FileNotFoundError("No Gymnasium policy was found. Run train-sim-parallel first.")

    if sim_config is None:
        handoff_root = Path(output_dir) if output_dir is not None else resolved_model_path.parent
        handoff = load_gym_handoff(handoff_root)
        if handoff is not None:
            sim_config = SimConfig(**handoff.recommended_sim_config)

    if env_factory is None:
        env = DinoSimEnv(config=sim_config or SimConfig(), render_mode=None)
    else:
        env = env_factory()

    frames = 0
    reward_total = 0.0
    passed_obstacles = 0
    episodes = 0
    start = time.time()

    try:
        obs, info = env.reset()
        while duration_s is None or time.time() - start < duration_s:
            action = int(policy.act_batch(np.asarray(obs, dtype=np.float32)[None, :])[0])
            obs, reward, terminated, truncated, info = env.step(action)
            reward_total += float(reward)
            passed_obstacles = max(passed_obstacles, int(info.get("passed_obstacles", 0)))
            frames += 1
            frame = env.render()
            if display and frame is not None:
                action_name = ("noop", "jump", "duck")[action]
                overlay = annotate_parallel_frame(
                    frame,
                    reward=reward_total,
                    passed_obstacles=passed_obstacles,
                    action=action_name,
                    episodes=episodes,
                    frames=frames,
                )
                cv2.imshow("DinoIA - Sim Parallel", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if terminated or truncated:
                episodes += 1
                obs, info = env.reset()

        return GymParallelRunResult(
            model_path=str(resolved_model_path),
            frames=frames,
            reward=reward_total,
            passed_obstacles=passed_obstacles,
            episodes=episodes,
        )
    finally:
        env.close()
        if display:
            cv2.destroyAllWindows()
