from __future__ import annotations

import numpy as np
import gymnasium as gym

from dinoia.config import SimConfig
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
