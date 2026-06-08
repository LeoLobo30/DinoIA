from __future__ import annotations

import argparse
from pathlib import Path

from .config import SimConfig
from .diagnostics import DEFAULT_DQN_ARTIFACTS_DIR, format_doctor_report, run_doctor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dinoia", description="DinoIA DQN vector-state simulator"
    )
    subparsers = parser.add_subparsers(dest="command")

    doctor_parser = subparsers.add_parser(
        "doctor", help="Check whether the DQN environment is ready"
    )
    doctor_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_DQN_ARTIFACTS_DIR
    )

    train_parser = subparsers.add_parser(
        "train-sim-dqn", help="Train DQN with vector observations in DinoEnv"
    )
    train_parser.add_argument("--total-timesteps", type=int, default=300_000)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--buffer-size", type=int, default=200_000)
    train_parser.add_argument("--learning-starts", type=int, default=5_000)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--n-quantiles", type=int, default=50)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--n-envs", type=int, default=1)
    train_parser.add_argument("--train-freq", type=int, default=4)
    train_parser.add_argument("--gradient-steps", type=int, default=1)
    train_parser.add_argument("--target-update-interval", type=int, default=10_000)
    train_parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    train_parser.add_argument("--exploration-fraction", type=float, default=0.25)
    train_parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    train_parser.add_argument("--eval-freq", type=int, default=10_000)
    train_parser.add_argument("--eval-episodes", type=int, default=20)
    train_parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    train_parser.add_argument(
        "--device", type=str, default="auto", help="cpu, cuda, cuda:0, gpu, or auto"
    )
    train_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_DQN_ARTIFACTS_DIR
    )
    train_parser.add_argument(
        "--source-model", type=Path, default=None, help="Resume DQN from a checkpoint"
    )
    _add_sim_args(train_parser)

    eval_parser = subparsers.add_parser(
        "eval-sim-dqn", help="Evaluate a DQN model in the simulator"
    )
    eval_parser.add_argument("--model-path", type=Path, default=None)
    eval_parser.add_argument(
        "--best", action="store_true", help="Use best_sim_model.zip if available"
    )
    eval_parser.add_argument("--episodes", type=int, default=10)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument(
        "--device", type=str, default="auto", help="cpu, cuda, cuda:0, gpu, or auto"
    )
    eval_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_DQN_ARTIFACTS_DIR
    )
    _add_sim_args(eval_parser)

    play_parser = subparsers.add_parser(
        "play-sim-dqn", help="Run a DQN model in DinoEnv"
    )
    play_parser.add_argument("--model-path", type=Path, default=None)
    play_parser.add_argument(
        "--best", action="store_true", help="Use best_sim_model.zip if available"
    )
    play_parser.add_argument("--duration", type=float, default=30.0)
    play_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_DQN_ARTIFACTS_DIR
    )
    play_parser.add_argument(
        "--device", type=str, default="auto", help="cpu, cuda, cuda:0, gpu, or auto"
    )
    play_parser.add_argument("--no-display", action="store_true")
    _add_sim_args(play_parser)

    real_play_parser = subparsers.add_parser(
        "play-real-dqn", help="Run a DQN model in the real Chrome Dino game"
    )
    real_play_parser.add_argument("--model-path", type=Path, default=None)
    real_play_parser.add_argument(
        "--best", action="store_true", help="Use best_sim_model.zip if available"
    )
    real_play_parser.add_argument("--duration", type=float, default=30.0)
    real_play_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_DQN_ARTIFACTS_DIR
    )
    real_play_parser.add_argument(
        "--device", type=str, default="auto", help="cpu, cuda, cuda:0, gpu, or auto"
    )
    real_play_parser.add_argument("--url", type=str, default="chrome://dino/")
    real_play_parser.add_argument("--chrome-binary", type=str, default=None)
    real_play_parser.add_argument("--headless", action="store_true")
    real_play_parser.add_argument("--window-width", type=int, default=1000)
    real_play_parser.add_argument("--window-height", type=int, default=320)
    real_play_parser.add_argument("--no-restart-on-crash", action="store_true")

    results_parser = subparsers.add_parser("results", help="Summarize DQN artifacts")
    results_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_DQN_ARTIFACTS_DIR
    )

    return parser


def _add_sim_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--action-latency-steps", type=int, default=0)
    parser.add_argument("--observation-latency-steps", type=int, default=0)
    parser.add_argument("--observation-noise", type=float, default=0.0)
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--domain-randomization-strength", type=float, default=0.12)


def _sim_config_from_args(args) -> SimConfig:
    return SimConfig(
        action_latency_steps=args.action_latency_steps,
        observation_latency_steps=args.observation_latency_steps,
        observation_noise=args.observation_noise,
        domain_randomization=args.domain_randomization,
        domain_randomization_strength=args.domain_randomization_strength,
        seed=getattr(args, "seed", 42),
    )


def run_train_sim_dqn(args):
    from .dqn_agent import train_sim_dqn

    result = train_sim_dqn(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        n_quantiles=args.n_quantiles,
        gamma=args.gamma,
        n_envs=args.n_envs,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        checkpoint_freq=args.checkpoint_freq,
        device=args.device,
        output_dir=args.output_dir,
        source_model=args.source_model,
        sim_config=_sim_config_from_args(args),
    )
    print(f"train-sim-dqn: model={result.model_path}")
    if result.best_model_path is not None:
        print(f"train-sim-dqn: best={result.best_model_path}")
    return result


def run_eval_sim_dqn(args):
    from .dqn_agent import evaluate_sim_dqn

    result = evaluate_sim_dqn(
        model_path=args.model_path,
        best=args.best,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        sim_config=_sim_config_from_args(args),
    )
    print(
        "eval-sim-dqn: episodes={episodes} mean_reward={reward:.3f} "
        "mean_passed={passed:.2f} max_passed={max_passed}".format(
            episodes=result.episodes,
            reward=result.mean_reward,
            passed=result.mean_passed_obstacles,
            max_passed=result.max_passed_obstacles,
        )
    )
    return result


def run_play_sim_dqn(args):
    from .dqn_agent import play_sim_dqn

    result = play_sim_dqn(
        model_path=args.model_path,
        best=args.best,
        duration=args.duration,
        output_dir=args.output_dir,
        device=args.device,
        no_display=args.no_display,
        sim_config=_sim_config_from_args(args),
    )
    print(
        "play-sim-dqn: steps={steps} reward={reward:.3f} passed={passed} episodes={episodes}".format(
            steps=result.steps,
            reward=result.reward,
            passed=result.passed_obstacles,
            episodes=result.episodes,
        )
    )
    return result


def run_play_real_dqn(args):
    from .real_game import play_real_dqn

    result = play_real_dqn(
        model_path=args.model_path,
        best=args.best,
        duration=args.duration,
        output_dir=args.output_dir,
        device=args.device,
        url=args.url,
        chrome_binary=args.chrome_binary,
        headless=args.headless,
        window_width=args.window_width,
        window_height=args.window_height,
        restart_on_crash=not args.no_restart_on_crash,
    )
    print(
        "play-real-dqn: steps={steps} crashes={crashes} passed={passed} episodes={episodes} url={url}".format(
            steps=result.steps,
            crashes=result.crashes,
            passed=result.passed_obstacles,
            episodes=result.episodes,
            url=result.url,
        )
    )
    return result


def run_results(args):
    from .dqn_agent import summarize_dqn

    summary = summarize_dqn(args.output_dir)
    print(f"results: output_dir={summary.output_dir}")
    print(f"results: model={summary.model_path or 'none'}")
    print(f"results: best_model={summary.best_model_path or 'none'}")
    print(f"results: training_runs={len(summary.history)}")
    print(f"results: eval_runs={len(summary.evaluation_history)}")
    latest = summary.latest_eval
    if latest is not None:
        print(
            "results: latest_eval mean_reward={reward} mean_passed={passed}".format(
                reward=latest.get("mean_reward", "none"),
                passed=latest.get("mean_passed_obstacles", "none"),
            )
        )
    best = summary.best_eval
    if best is not None:
        print(
            "results: best_eval mean_reward={reward} mean_passed={passed}".format(
                reward=best.get("mean_reward", "none"),
                passed=best.get("mean_passed_obstacles", "none"),
            )
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["doctor"])

    if args.command == "doctor":
        report = run_doctor(args.output_dir)
        print(format_doctor_report(report))
        return 0 if report.ok else 1
    if args.command == "train-sim-dqn":
        run_train_sim_dqn(args)
        return 0
    if args.command == "eval-sim-dqn":
        run_eval_sim_dqn(args)
        return 0
    if args.command == "play-sim-dqn":
        run_play_sim_dqn(args)
        return 0
    if args.command == "play-real-dqn":
        run_play_real_dqn(args)
        return 0
    if args.command == "results":
        run_results(args)
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2
