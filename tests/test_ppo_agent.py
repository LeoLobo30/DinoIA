from __future__ import annotations

import json
import uuid
from pathlib import Path

import gymnasium as gym
import numpy as np

from dinoia import ppo_agent
from dinoia.config import RealGameConfig
from dinoia.real_env import RealEnvConfig


class FakePPO:
    loaded = []

    def __init__(self, policy=None, env=None, **kwargs):
        self.policy = policy
        self.env = env
        self.kwargs = kwargs
        self.learn_calls = []

    @classmethod
    def load(cls, path, env=None, device="auto", custom_objects=None):
        cls.loaded.append({"path": path, "env": env, "device": device, "custom_objects": custom_objects})
        return cls(policy="loaded", env=env, device=device)

    def learn(self, total_timesteps, reset_num_timesteps=True):
        self.learn_calls.append(
            {
                "total_timesteps": total_timesteps,
                "reset_num_timesteps": reset_num_timesteps,
            }
        )
        return self

    def save(self, path):
        path.write_text("fake-ppo-model", encoding="utf-8") if hasattr(path, "write_text") else None
        if not hasattr(path, "write_text"):
            from pathlib import Path

            Path(path).write_text("fake-ppo-model", encoding="utf-8")


class FakeRealEnv(gym.Env):
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(13,), dtype=np.float32)
        FakeRealEnv.instances.append(self)

    def reset(self, *, seed=None, options=None):
        return np.zeros((13,), dtype=np.float32), {}

    def step(self, action):
        return np.zeros((13,), dtype=np.float32), 0.0, True, False, {"passed_obstacles": 0}

    def close(self):
        self.closed = True


def test_train_real_ppo_loads_sim_checkpoint_before_learning(monkeypatch):
    FakePPO.loaded = []
    FakeRealEnv.instances = []
    monkeypatch.setattr(ppo_agent, "_load_ppo", lambda: FakePPO)
    monkeypatch.setattr(ppo_agent, "DinoRealEnv", FakeRealEnv)
    monkeypatch.setattr(
        ppo_agent,
        "evaluate_sim_ppo",
        lambda **kwargs: ppo_agent.PPOEvalResult(
            model_path=Path(kwargs["model_path"]),
            run_dir=Path(kwargs.get("output_dir", "artifacts")),
            episodes=1,
            steps=1,
            mean_reward=1.0,
            reward_std=0.0,
            mean_passed_obstacles=1.0,
            max_passed_obstacles=1,
            action_counts={"noop": 1, "jump": 0, "duck": 0},
            episode_rewards=[1.0],
            episode_lengths=[1],
        ),
    )

    output_dir = Path("artifacts") / f"test-ppo-{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_model = output_dir / "sim_model.zip"
    source_model.write_text("fake-source", encoding="utf-8")

    result = ppo_agent.train_real_ppo(
        total_timesteps=8,
        output_dir=output_dir,
        source_model_path=source_model,
        real_config=RealGameConfig(debug=False, focus_window=False),
        env_config=RealEnvConfig(step_interval_s=0.0, reset_wait_s=0.0, enable_action_gate=False),
        device="cpu",
    )

    assert FakePPO.loaded[0]["path"] == str(source_model)
    assert FakePPO.loaded[0]["device"] == "cpu"
    assert result.source_model_path == source_model
    assert result.model_path == output_dir / "real_model.zip"
    assert result.model_path.exists()
    assert FakeRealEnv.instances[0].closed is True

    history = json.loads((output_dir / "training_history.json").read_text(encoding="utf-8"))
    assert history[-1]["stage"] == "real"
    assert history[-1]["source_model_path"] == str(source_model)


def test_train_real_ppo_defaults_to_best_sim_checkpoint(monkeypatch):
    FakePPO.loaded = []
    FakeRealEnv.instances = []
    monkeypatch.setattr(ppo_agent, "_load_ppo", lambda: FakePPO)
    monkeypatch.setattr(ppo_agent, "DinoRealEnv", FakeRealEnv)
    monkeypatch.setattr(
        ppo_agent,
        "evaluate_sim_ppo",
        lambda **kwargs: ppo_agent.PPOEvalResult(
            model_path=Path(kwargs["model_path"]),
            run_dir=Path(kwargs.get("output_dir", "artifacts")),
            episodes=1,
            steps=1,
            mean_reward=1.0,
            reward_std=0.0,
            mean_passed_obstacles=1.0,
            max_passed_obstacles=1,
            action_counts={"noop": 1, "jump": 0, "duck": 0},
            episode_rewards=[1.0],
            episode_lengths=[1],
        ),
    )

    output_dir = Path("artifacts") / f"test-ppo-{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_source = output_dir / "best_sim_model.zip"
    best_source.write_text("best-source", encoding="utf-8")

    ppo_agent._write_json(output_dir / "best_sim_eval.json", {"mean_reward": 1.0, "mean_passed_obstacles": 1.0})
    ppo_agent._write_json(output_dir / "best_metrics.json", {"mean_reward": 1.0, "mean_passed_obstacles": 1.0})

    result = ppo_agent.train_real_ppo(
        total_timesteps=4,
        output_dir=output_dir,
        real_config=RealGameConfig(debug=False, focus_window=False),
        env_config=RealEnvConfig(step_interval_s=0.0, reset_wait_s=0.0, enable_action_gate=False),
        device="cpu",
    )

    assert FakePPO.loaded[0]["path"] == str(best_source)
    assert result.source_model_path == best_source


def test_update_best_metrics_only_promotes_when_improved():
    output_dir = Path("artifacts") / f"best-metrics-{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = ppo_agent.PPOArtifacts(output_dir=output_dir)
    source = output_dir / "candidate.zip"
    source.write_text("candidate", encoding="utf-8")
    ppo_agent._write_json(
        artifacts.best_sim_eval_path,
        {"mean_reward": 5.0, "mean_passed_obstacles": 3.0, "model_path": str(output_dir / "old.zip")},
    )
    best_model = output_dir / "best_sim_model.zip"
    best_model.write_text("old-best", encoding="utf-8")

    promoted = ppo_agent._update_best_metrics(
        artifacts,
        stage="sim",
        metrics={"mean_reward": 4.0, "mean_passed_obstacles": 2.0},
        model_path=source,
    )

    assert promoted is False
    assert best_model.read_text(encoding="utf-8") == "old-best"


def test_evaluate_sim_ppo_writes_metrics(monkeypatch):
    class FakeEvalEnv(gym.Env):
        def __init__(self, config=None):
            self.calls = 0
            self.action_space = gym.spaces.Discrete(3)
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(13,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            self.calls = 0
            return np.zeros((13,), dtype=np.float32), {}

        def step(self, action):
            self.calls += 1
            done = self.calls >= 2
            return np.zeros((13,), dtype=np.float32), 1.0, done, False, {"passed_obstacles": self.calls}

        def close(self):
            return None

    fake_model = FakePPO(policy="loaded", env=None)
    fake_model.predict = lambda obs, deterministic=True: (0, None)

    monkeypatch.setattr(ppo_agent, "_load_ppo", lambda: FakePPO)
    monkeypatch.setattr(FakePPO, "load", classmethod(lambda cls, path, env=None, device="auto", custom_objects=None: fake_model))
    monkeypatch.setattr(ppo_agent, "DinoEnv", FakeEvalEnv)

    output_dir = Path("artifacts") / f"eval-ppo-{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "sim_model.zip"
    model_path.write_text("fake", encoding="utf-8")

    result = ppo_agent.evaluate_sim_ppo(model_path=model_path, output_dir=output_dir, episodes=2, device="cpu")

    assert result.episodes == 2
    assert result.mean_reward > 0
    assert (output_dir / "latest_eval.json").exists()


def test_train_sim_ppo_records_transfer_eval(monkeypatch):
    class FakeEvalResult:
        def __init__(self, mean_reward, mean_passed_obstacles, episodes=1):
            self.run_dir = Path("artifacts") / "fake-eval"
            self.episodes = episodes
            self.steps = episodes
            self.mean_reward = mean_reward
            self.reward_std = 0.0
            self.mean_passed_obstacles = mean_passed_obstacles
            self.max_passed_obstacles = int(mean_passed_obstacles)
            self.action_counts = {"noop": 1, "jump": 0, "duck": 0}
            self.episode_rewards = [mean_reward]
            self.episode_lengths = [episodes]

    class FakeEnv(gym.Env):
        def __init__(self, config=None):
            self.config = config
            self.action_space = gym.spaces.Discrete(3)
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(13,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros((13,), dtype=np.float32), {}

        def step(self, action):
            return np.zeros((13,), dtype=np.float32), 1.0, True, False, {"passed_obstacles": 1}

        def close(self):
            return None

    class FakeModel:
        def __init__(self):
            self.env = None

        def set_env(self, env):
            self.env = env

        def learn(self, total_timesteps, reset_num_timesteps=True):
            return self

        def save(self, path):
            Path(path).write_text("fake", encoding="utf-8")

    class FakePPO:
        def __init__(self, policy=None, env=None, **kwargs):
            self.policy = policy
            self.env = env
            self.kwargs = kwargs
            self.learn_calls = []

        @classmethod
        def load(cls, path, env=None, device="cpu", custom_objects=None):
            return FakeModel()

        def set_env(self, env):
            self.env = env

        def learn(self, total_timesteps, reset_num_timesteps=True):
            self.learn_calls.append(
                {
                    "total_timesteps": total_timesteps,
                    "reset_num_timesteps": reset_num_timesteps,
                }
            )
            return self

        def save(self, path):
            Path(path).write_text("fake", encoding="utf-8")

    eval_calls = []

    monkeypatch.setattr(ppo_agent, "DinoEnv", FakeEnv)
    monkeypatch.setattr(ppo_agent, "_load_ppo", lambda: FakePPO)

    def fake_eval(**kwargs):
        eval_calls.append(kwargs.get("sim_config"))
        sim_config = kwargs.get("sim_config")
        if sim_config is not None and getattr(sim_config, "domain_randomization", False):
            return ppo_agent.PPOEvalResult(
                model_path=Path(kwargs["model_path"]),
                run_dir=Path("artifacts") / "eval",
                episodes=1,
                steps=1,
                mean_reward=2.0,
                reward_std=0.0,
                mean_passed_obstacles=2.0,
                max_passed_obstacles=2,
                action_counts={"noop": 1, "jump": 0, "duck": 0},
                episode_rewards=[2.0],
                episode_lengths=[1],
            )
        return ppo_agent.PPOEvalResult(
            model_path=Path(kwargs["model_path"]),
            run_dir=Path("artifacts") / "eval",
            episodes=1,
            steps=1,
            mean_reward=1.0,
            reward_std=0.0,
            mean_passed_obstacles=1.0,
            max_passed_obstacles=1,
            action_counts={"noop": 1, "jump": 0, "duck": 0},
            episode_rewards=[1.0],
            episode_lengths=[1],
        )

    monkeypatch.setattr(ppo_agent, "evaluate_sim_ppo", fake_eval)

    output_dir = Path("artifacts") / f"train-sim-{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = ppo_agent.train_sim_ppo(total_timesteps=6, output_dir=output_dir, device="cpu", verbose=0)

    assert result.source_model_path is None
    history = json.loads((output_dir / "training_history.json").read_text(encoding="utf-8"))
    assert history[-1]["best_model_path"] is not None
    assert any(getattr(cfg, "domain_randomization", False) for cfg in eval_calls if cfg is not None)
