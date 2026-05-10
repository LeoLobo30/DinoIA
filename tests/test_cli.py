from __future__ import annotations

from dinoia.cli import build_parser


def test_cli_parses_evolve_real_defaults():
    args = build_parser().parse_args(["evolve-real"])

    assert args.command == "evolve-real"
    assert args.generations == 20
    assert args.population == 24


def test_cli_parses_evolve_real_resume_options():
    args = build_parser().parse_args(
        ["evolve-real", "--resume", "--resume-checkpoint", "artifacts/neat/neat-checkpoint-7"]
    )

    assert args.command == "evolve-real"
    assert args.resume is True
    assert str(args.resume_checkpoint).endswith("neat-checkpoint-7")


def test_cli_parses_play_real_neat():
    args = build_parser().parse_args(["play-real-neat", "--best", "--duration", "5"])

    assert args.command == "play-real-neat"
    assert args.best is True
    assert args.duration == 5.0


def test_cli_parses_results():
    args = build_parser().parse_args(["results"])

    assert args.command == "results"


def test_cli_parses_train_sim_ppo_defaults():
    args = build_parser().parse_args(["train-sim-ppo"])

    assert args.command == "train-sim-ppo"
    assert args.total_timesteps == 250_000
    assert args.action_latency_steps == 1
    assert args.observation_latency_steps == 1
    assert args.observation_noise == 0.03
    assert args.domain_randomization_strength == 0.18
    assert args.device == "cpu"


def test_cli_parses_eval_sim_ppo():
    args = build_parser().parse_args(["eval-sim-ppo", "--best", "--episodes", "8"])

    assert args.command == "eval-sim-ppo"
    assert args.best is True
    assert args.episodes == 8
    assert args.device == "cpu"


def test_cli_parses_train_real_ppo_source_model():
    args = build_parser().parse_args(
        ["train-real-ppo", "--source-model", "artifacts/ppo/sim_model.zip", "--total-timesteps", "64"]
    )

    assert args.command == "train-real-ppo"
    assert str(args.source_model).endswith("sim_model.zip")
    assert args.total_timesteps == 64
    assert args.device == "cpu"


def test_cli_parses_play_sim_ppo():
    args = build_parser().parse_args(["play-sim-ppo", "--best", "--duration", "7"])

    assert args.command == "play-sim-ppo"
    assert args.best is True
    assert args.duration == 7.0
    assert args.device == "cpu"


def test_cli_parses_play_real_ppo():
    args = build_parser().parse_args(["play-real-ppo", "--best", "--duration", "5"])

    assert args.command == "play-real-ppo"
    assert args.best is True
    assert args.duration == 5.0
    assert args.device == "cpu"


def test_cli_parses_train_sim_parallel():
    args = build_parser().parse_args(["train-sim-parallel", "--generations", "3", "--workers", "2"])

    assert args.command == "train-sim-parallel"
    assert args.generations == 3
    assert args.workers == 2


def test_cli_parses_train_sim_parallel_transfer_defaults():
    args = build_parser().parse_args(["train-sim-parallel"])

    assert args.command == "train-sim-parallel"
    assert args.action_latency_steps == 1
    assert args.observation_latency_steps == 1
    assert args.observation_noise == 0.03
    assert args.domain_randomization_strength == 0.18


def test_cli_parses_play_sim_parallel():
    args = build_parser().parse_args(["play-sim-parallel", "--best", "--duration", "7"])

    assert args.command == "play-sim-parallel"
    assert args.best is True
    assert args.duration == 7.0
