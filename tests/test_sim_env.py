from __future__ import annotations

import numpy as np
import gymnasium as gym
import pytest

from dinoia.config import SimConfig
from dinoia.dqn_agent import _clean_eval_config
from dinoia.sim import ENV_ID, DinoEnv
from dinoia.sim.env import DinoSimEnv


def test_jump_cycle_and_observation_shape():
    env = DinoSimEnv(
        config=SimConfig(min_spawn_gap_px=9999, max_spawn_gap_px=10000, domain_randomization=False, observation_noise=0.0),
        render_mode=None,
    )
    obs, info = env.reset(seed=123)

    assert obs.shape == (13,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)

    start_y = env.player_y
    obs, reward, terminated, truncated, info = env.step(1)
    assert env.player_y < start_y
    assert not terminated
    assert not truncated

    for _ in range(240):
        obs, reward, terminated, truncated, info = env.step(0)
        if env.grounded:
            break

    assert env.grounded
    assert env.player_y == env.floor_y - env.config.stand_height


def test_collision_terminates_episode():
    env = DinoSimEnv(
        config=SimConfig(min_spawn_gap_px=9999, max_spawn_gap_px=10000, domain_randomization=False, observation_noise=0.0),
        render_mode=None,
    )
    env.reset(seed=321)
    env._obstacles.clear()

    env.spawn_obstacle(
        kind="cactus",
        x=env.config.player_x + env.config.player_width - 2,
        width=22,
        height=40,
    )

    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated
    assert reward < 0
    assert info["collision"] is True


def test_action_latency_delays_jump_execution():
    env = DinoSimEnv(
        config=SimConfig(
            min_spawn_gap_px=9999,
            max_spawn_gap_px=10000,
            domain_randomization=False,
            observation_noise=0.0,
            action_latency_steps=1,
        ),
        render_mode=None,
    )
    env.reset(seed=123)

    start_y = env.player_y
    env.step(1)
    assert env.player_y == start_y

    env.step(0)
    assert env.player_y < start_y


def test_unnecessary_duck_is_penalized():
    env = DinoSimEnv(
        config=SimConfig(
            min_spawn_gap_px=9999,
            max_spawn_gap_px=10000,
            domain_randomization=False,
            observation_noise=0.0,
            action_latency_steps=0,
            observation_latency_steps=0,
        ),
        render_mode=None,
    )
    env.reset(seed=123)

    _, reward, terminated, truncated, info = env.step(2)

    assert terminated is False
    assert truncated is False
    assert reward < env.config.survival_reward
    assert info["requested_action"] == 2
    assert info["executed_action"] == 2


def test_initial_obstacle_delay_keeps_start_clear():
    env = DinoSimEnv(
        config=SimConfig(
            initial_obstacle_delay_s=2.0,
            min_spawn_gap_px=1,
            max_spawn_gap_px=1,
            domain_randomization=False,
            observation_noise=0.0,
            action_latency_steps=0,
            observation_latency_steps=0,
        ),
        render_mode=None,
    )
    env.reset(seed=123)

    for _ in range(40):
        env.step(0)
        assert len(env._obstacles) == 0


def test_dino_env_registered_with_gymnasium():
    env = gym.make(ENV_ID)
    try:
        obs, info = env.reset(seed=123)

        assert isinstance(env.unwrapped, DinoEnv)
        assert obs.shape == (13,)
        assert info["passed_obstacles"] == 0
    finally:
        env.close()


def test_curriculum_speed_phases_progress_by_episode():
    env = DinoSimEnv(
        config=SimConfig(
            domain_randomization=False,
            curriculum_learning=True,
            curriculum_low_episodes=1,
            curriculum_medium_episodes=2,
            curriculum_high_episodes=3,
            curriculum_low_speed_min=360.0,
            curriculum_low_speed_max=360.0,
            curriculum_medium_speed_min=480.0,
            curriculum_medium_speed_max=480.0,
            curriculum_high_speed_min=660.0,
            curriculum_high_speed_max=660.0,
            curriculum_random_speed_min=900.0,
            curriculum_random_speed_max=900.0,
        ),
        render_mode=None,
    )
    try:
        env.reset(seed=123)
        assert env.current_speed == 360.0
        env.reset(seed=123)
        assert env.current_speed == 480.0
        env.reset(seed=123)
        assert env.current_speed == 660.0
        env.reset(seed=123)
        assert env.current_speed == 900.0
    finally:
        env.close()


def test_pass_reward_scales_with_speed():
    env = DinoSimEnv(
        config=SimConfig(
            domain_randomization=False,
            curriculum_learning=False,
            observation_noise=0.0,
            base_speed=600.0,
            max_speed=600.0,
            speed_acceleration=0.0,
            pass_reward=1.0,
            speed_pass_reward_scale=0.1,
            speed_unit_px_s=60.0,
            min_spawn_gap_px=9999,
            max_spawn_gap_px=10000,
        ),
        render_mode=None,
    )
    try:
        env.reset(seed=123)
        env._obstacles.clear()
        env.spawn_obstacle(kind="cactus", x=0.0, width=10, height=40)

        _, reward, terminated, truncated, info = env.step(0)

        assert not terminated
        assert not truncated
        assert info["passed_obstacles"] == 1
        assert reward == pytest.approx(env.config.survival_reward + 2.0)
    finally:
        env.close()


def test_clean_eval_config_disables_training_noise():
    train_config = SimConfig(
        domain_randomization=True,
        observation_noise=0.2,
        action_latency_steps=2,
        observation_latency_steps=3,
        curriculum_learning=True,
    )

    eval_config = _clean_eval_config(train_config)

    assert eval_config.domain_randomization is False
    assert eval_config.observation_noise == 0.0
    assert eval_config.action_latency_steps == 0
    assert eval_config.observation_latency_steps == 0
    assert eval_config.curriculum_learning is False
