from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "neat"


@dataclass(slots=True)
class NeatProjectResults:
    output_dir: Path
    winner_path: Path | None
    best_genome_path: Path | None
    latest_checkpoint_path: Path | None
    neat_config_path: Path | None
    history_path: Path | None
    summary_path: Path | None
    history: list[dict[str, Any]]

    @property
    def best_fitness(self) -> float | None:
        if not self.history:
            return None
        return max(float(item.get("best_fitness", 0.0)) for item in self.history)

    @property
    def best_generation(self) -> int | None:
        if not self.history:
            return None
        best = max(self.history, key=lambda item: float(item.get("best_fitness", 0.0)))
        generation = best.get("generation")
        return int(generation) if generation is not None else None

    @property
    def best_distance_px(self) -> float | None:
        if not self.history:
            return None
        best = max(self.history, key=lambda item: float(item.get("best_fitness", 0.0)))
        value = best.get("best_distance_px")
        return float(value) if value is not None else None

    @property
    def latest_mean_fitness(self) -> float | None:
        if not self.history:
            return None
        value = self.history[-1].get("mean_fitness")
        return float(value) if value is not None else None


def artifacts_dir(output_dir: str | Path | None = None) -> Path:
    return Path(output_dir) if output_dir is not None else DEFAULT_ARTIFACTS_DIR


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_training_history(output_dir: str | Path | None = None) -> list[dict[str, Any]]:
    raw = read_json(artifacts_dir(output_dir) / "training_history.json")
    return raw if isinstance(raw, list) else []


def list_neat_checkpoints(output_dir: str | Path | None = None) -> list[Path]:
    root = artifacts_dir(output_dir)
    checkpoints = sorted(
        root.glob("neat-checkpoint-*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return checkpoints


def select_latest_model(output_dir: str | Path | None = None) -> Path | None:
    root = artifacts_dir(output_dir)
    winner = root / "winner.pkl"
    if winner.exists():
        return winner
    best = root / "best_genome.pkl"
    if best.exists():
        return best
    checkpoints = list_neat_checkpoints(root)
    return checkpoints[0] if checkpoints else None


def select_best_model(output_dir: str | Path | None = None) -> Path | None:
    return select_latest_model(output_dir)


def summarize_project(output_dir: str | Path | None = None) -> NeatProjectResults:
    root = artifacts_dir(output_dir)
    checkpoints = list_neat_checkpoints(root)
    winner_path = root / "winner.pkl"
    best_genome_path = root / "best_genome.pkl"
    neat_config_path = root / "neat_config.txt"
    history_path = root / "training_history.json"
    summary_path = root / "latest_summary.txt"
    return NeatProjectResults(
        output_dir=root,
        winner_path=winner_path if winner_path.exists() else None,
        best_genome_path=best_genome_path if best_genome_path.exists() else None,
        latest_checkpoint_path=checkpoints[0] if checkpoints else None,
        neat_config_path=neat_config_path if neat_config_path.exists() else None,
        history_path=history_path if history_path.exists() else None,
        summary_path=summary_path if summary_path.exists() else None,
        history=read_training_history(root),
    )
