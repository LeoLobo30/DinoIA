from __future__ import annotations

import numpy as np

from dinoia.config import SimConfig
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
