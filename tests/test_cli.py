from __future__ import annotations

from dinoia.cli import build_parser


def test_cli_parses_train_sim_dqn_defaults():
    args = build_parser().parse_args(["train-sim-dqn"])

    assert args.command == "train-sim-dqn"
    assert args.total_timesteps == 300_000
    assert args.learning_rate == 1e-4
    assert args.buffer_size == 200_000
    assert args.learning_starts == 5_000
    assert args.batch_size == 64
    assert args.n_quantiles == 50
    assert args.n_envs == 1
    assert args.train_freq == 4
    assert args.gradient_steps == 1
    assert args.target_update_interval == 10_000
    assert args.exploration_initial_eps == 1.0
    assert args.exploration_fraction == 0.25
    assert args.exploration_final_eps == 0.05
    assert args.eval_freq == 10_000
    assert args.eval_episodes == 20
    assert args.checkpoint_freq == 50_000
    assert args.device == "auto"
    assert args.action_latency_steps == 0
    assert args.observation_latency_steps == 0
    assert args.observation_noise == 0.0
    assert args.domain_randomization is False
    assert args.domain_randomization_strength == 0.12


def test_cli_parses_eval_sim_dqn():
    args = build_parser().parse_args(["eval-sim-dqn", "--best", "--episodes", "8"])

    assert args.command == "eval-sim-dqn"
    assert args.best is True
    assert args.episodes == 8
    assert args.device == "auto"


def test_cli_parses_play_sim_dqn():
    args = build_parser().parse_args(["play-sim-dqn", "--best", "--duration", "7", "--no-display"])

    assert args.command == "play-sim-dqn"
    assert args.best is True
    assert args.duration == 7.0
    assert args.no_display is True
    assert args.device == "auto"


def test_cli_parses_results():
    args = build_parser().parse_args(["results"])

    assert args.command == "results"
