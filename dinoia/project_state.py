from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "dqn"


@dataclass(slots=True)
class ModelCandidate:
    path: Path
    kind: str
    step: int | None = None
    modified_at: float | None = None


@dataclass(slots=True)
class ProjectResults:
    output_dir: Path
    training_config_path: Path | None
    model_path: Path | None
    best_model_path: Path | None
    latest_checkpoint_path: Path | None
    evaluation_path: Path | None
    training_config: dict[str, Any] | None
    evaluation: dict[str, Any] | None
    checkpoints: list[ModelCandidate]


_STEP_PATTERN = re.compile(r"(?P<step>\d+)_steps", re.IGNORECASE)


def artifacts_dir(output_dir: str | Path | None = None) -> Path:
    return Path(output_dir) if output_dir is not None else DEFAULT_ARTIFACTS_DIR


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_training_config(output_dir: str | Path | None = None) -> dict[str, Any] | None:
    return read_json(artifacts_dir(output_dir) / "training_config.json")


def read_evaluation(output_dir: str | Path | None = None) -> dict[str, Any] | None:
    evaluation_path = artifacts_dir(output_dir) / "eval" / "evaluations.npz"
    if not evaluation_path.exists():
        return None

    with np.load(evaluation_path, allow_pickle=False) as data:
        rewards = data["results"].astype(float).ravel().tolist() if "results" in data else []
        lengths = data["ep_lengths"].astype(int).ravel().tolist() if "ep_lengths" in data else []
        timesteps = data["timesteps"].astype(int).ravel().tolist() if "timesteps" in data else []

    return {
        "path": evaluation_path,
        "timesteps": timesteps,
        "rewards": rewards,
        "lengths": lengths,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "std_length": float(np.std(lengths)) if lengths else 0.0,
    }


def _checkpoint_step(path: Path) -> int | None:
    match = _STEP_PATTERN.search(path.stem)
    if match is None:
        return None
    return int(match.group("step"))


def list_checkpoints(output_dir: str | Path | None = None) -> list[ModelCandidate]:
    checkpoint_dir = artifacts_dir(output_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        return []

    candidates: list[ModelCandidate] = []
    for path in checkpoint_dir.glob("*.zip"):
        candidates.append(
            ModelCandidate(
                path=path,
                kind="checkpoint",
                step=_checkpoint_step(path),
                modified_at=path.stat().st_mtime,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.step is None,
            candidate.step if candidate.step is not None else -1,
            candidate.modified_at if candidate.modified_at is not None else 0.0,
        ),
        reverse=True,
    )
    return candidates


def find_model_candidates(output_dir: str | Path | None = None) -> list[ModelCandidate]:
    root = artifacts_dir(output_dir)
    candidates: list[ModelCandidate] = []

    final_model = root / "dino_dqn_final.zip"
    if final_model.exists():
        candidates.append(
            ModelCandidate(path=final_model, kind="final", modified_at=final_model.stat().st_mtime)
        )

    best_model = root / "best_model" / "best_model.zip"
    if best_model.exists():
        candidates.append(
            ModelCandidate(path=best_model, kind="best", modified_at=best_model.stat().st_mtime)
        )

    candidates.extend(list_checkpoints(root))
    return candidates


def select_latest_model(output_dir: str | Path | None = None) -> Path | None:
    root = artifacts_dir(output_dir)
    final_model = root / "dino_dqn_final.zip"
    if final_model.exists():
        return final_model

    best_model = root / "best_model" / "best_model.zip"
    if best_model.exists():
        return best_model

    checkpoints = list_checkpoints(root)
    return checkpoints[0].path if checkpoints else None


def select_best_model(output_dir: str | Path | None = None) -> Path | None:
    root = artifacts_dir(output_dir)
    best = root / "best_model" / "best_model.zip"
    if best.exists():
        return best
    return select_latest_model(root)


def _zip_contains_observation_size(path: Path) -> int | None:
    try:
        with zipfile.ZipFile(path) as archive:
            if "training_config.json" not in archive.namelist():
                return None
            with archive.open("training_config.json") as handle:
                raw = json.loads(handle.read().decode("utf-8"))
    except Exception:
        return None

    training = raw.get("training", raw)
    return training.get("observation_size")


def describe_resume_candidates(
    output_dir: str | Path | None = None,
    *,
    expected_observation_size: int | None = None,
) -> tuple[Path | None, list[str]]:
    notes: list[str] = []
    for candidate in find_model_candidates(output_dir):
        obs_size = _zip_contains_observation_size(candidate.path)
        if expected_observation_size is not None and obs_size is not None and obs_size != expected_observation_size:
            notes.append(
                f"skip {candidate.path.name}: observation_size={obs_size} expected={expected_observation_size}"
            )
            continue
        if expected_observation_size is not None and obs_size is None:
            notes.append(f"try {candidate.path.name}: no observation_size metadata")
        return candidate.path, notes

    return None, notes


def summarize_project(output_dir: str | Path | None = None) -> ProjectResults:
    root = artifacts_dir(output_dir)
    checkpoints = list_checkpoints(root)
    training_config_path = root / "training_config.json"
    best_model_path = root / "best_model" / "best_model.zip"
    latest_checkpoint_path = checkpoints[0].path if checkpoints else None
    model_path = select_latest_model(root)
    evaluation_path = root / "eval" / "evaluations.npz"

    return ProjectResults(
        output_dir=root,
        training_config_path=training_config_path if training_config_path.exists() else None,
        model_path=model_path,
        best_model_path=best_model_path if best_model_path.exists() else None,
        latest_checkpoint_path=latest_checkpoint_path,
        evaluation_path=evaluation_path if evaluation_path.exists() else None,
        training_config=read_training_config(root),
        evaluation=read_evaluation(root),
        checkpoints=checkpoints,
    )
