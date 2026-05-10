from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
from gymnasium.vector import SyncVectorEnv

from dinoia.config import SimConfig
from dinoia.gym_parallel import (
    GymParallelConfig,
    GymParallelTrainer,
    LinearPolicy,
    load_gym_handoff,
    annotate_parallel_frame,
    visualize_parallel_policy,
)


def test_linear_policy_maps_logits_to_actions():
    policy = LinearPolicy(
        weights=np.array(
            [
                [3.0] + [0.0] * 12,
                [0.0, 3.0] + [0.0] * 11,
                [0.0, 0.0, 3.0] + [0.0] * 10,
            ],
            dtype=np.float32,
        ),
        bias=np.zeros(3, dtype=np.float32),
    )
    obs = np.zeros((3, 13), dtype=np.float32)
    obs[0, 0] = 1.0
    obs[1, 1] = 1.0
    obs[2, 2] = 1.0

    actions = policy.act_batch(obs)

    assert actions.tolist() == [0, 1, 2]


def test_parallel_gym_training_writes_artifacts():
    output_dir = Path("artifacts") / f"gym-parallel-{uuid.uuid4().hex}"
    config = GymParallelConfig(
        generations=1,
        population=4,
        workers=2,
        rollout_steps=4,
        output_dir=output_dir,
        sim_config=SimConfig(
            min_spawn_gap_px=9999,
            max_spawn_gap_px=10000,
            domain_randomization=False,
            observation_noise=0.0,
        ),
    )

    trainer = GymParallelTrainer(config=config, vector_env_cls=SyncVectorEnv)
    policy_path = trainer.train()

    history_path = output_dir / "training_history.json"
    summary_path = output_dir / "latest_summary.txt"

    assert policy_path.exists()
    assert history_path.exists()
    assert summary_path.exists()

    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert len(history) == 1
    assert history[0]["best_reward"] is not None
    assert history[0]["best_policy_path"] == str(policy_path)
    assert (output_dir / "handoff.json").exists()

    handoff = load_gym_handoff(output_dir)
    assert handoff is not None
    assert handoff.source_policy_path == str(policy_path)
    assert "teacher_source" in handoff.recommended_neat_config


class FakeVisualEnv:
    def __init__(self):
        self.render_mode = "human"
        self._step = 0
        self.render_calls = 0

    def reset(self, *, seed=None, options=None):
        self._step = 0
        obs = np.zeros(13, dtype=np.float32)
        obs[0] = 0.1
        return obs, {"passed_obstacles": 0}

    def step(self, action):
        self._step += 1
        obs = np.zeros(13, dtype=np.float32)
        obs[action % 3] = 1.0
        reward = 1.0
        terminated = self._step >= 2
        truncated = False
        info = {"passed_obstacles": self._step}
        return obs, reward, terminated, truncated, info

    def render(self):
        self.render_calls += 1
        return None

    def close(self):
        return None


def test_visualize_parallel_policy_uses_saved_policy():
    output_dir = Path("artifacts") / f"gym-parallel-visual-{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "best_policy.npz"
    policy = LinearPolicy(
        weights=np.array(
            [
                [3.0] + [0.0] * 12,
                [0.0, 3.0] + [0.0] * 11,
                [0.0, 0.0, 3.0] + [0.0] * 10,
            ],
            dtype=np.float32,
        ),
        bias=np.zeros(3, dtype=np.float32),
    )
    np.savez_compressed(policy_path, weights=policy.weights, bias=policy.bias)

    env = FakeVisualEnv()
    result = visualize_parallel_policy(
        model_path=policy_path,
        duration_s=0.01,
        env_factory=lambda: env,
        display=False,
    )

    assert result.model_path == str(policy_path)
    assert result.frames > 0
    assert env.render_calls > 0


def test_annotate_parallel_frame_writes_overlay_text():
    frame = np.zeros((120, 240, 3), dtype=np.uint8)

    annotated = annotate_parallel_frame(
        frame,
        reward=3.25,
        passed_obstacles=2,
        action="jump",
        episodes=1,
        frames=42,
    )

    assert annotated.shape == frame.shape
    assert np.any(annotated != frame)
