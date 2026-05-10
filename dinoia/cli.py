from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import DinoVisionConfig, NeatConfig, RealGameConfig, SimConfig
from .diagnostics import format_doctor_report, run_doctor as run_environment_doctor
from .gym_parallel import GymParallelConfig, load_gym_handoff, train_parallel_gymnasium, visualize_parallel_policy
from .neat_agent import DinoRealNeatRunner, evolve_real_neat
from .ppo_agent import (
    DEFAULT_PPO_ARTIFACTS_DIR,
    evaluate_sim_ppo as evaluate_sim_ppo_agent,
    play_real_ppo as play_real_ppo_agent,
    play_sim_ppo as play_sim_ppo_agent,
    summarize_ppo,
    train_real_ppo as train_real_ppo_agent,
    train_sim_ppo as train_sim_ppo_agent,
)
from .project_state import DEFAULT_ARTIFACTS_DIR, list_neat_checkpoints, select_best_model, summarize_project
from .real_env import RealEnvConfig
from .real_game import DinoRealGameRunner
from .real_rl import RealRLObservationBuilder

DEFAULT_MANUAL_REGION = (1300, 70, 1200, 400)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dinoia", description="DinoIA PPO numeric-observation project")
    subparsers = parser.add_subparsers(dest="command")

    doctor_parser = subparsers.add_parser("doctor", help="Check whether the PPO environment is ready")
    doctor_parser.add_argument("--output-dir", type=Path, default=DEFAULT_PPO_ARTIFACTS_DIR)

    play_parser = subparsers.add_parser("play-real", help="Run the Chrome Dino heuristic vision agent")
    _add_real_args(play_parser)

    diagnose_parser = subparsers.add_parser("diagnose-real", help="Inspect real-game capture and numeric observation")
    _add_real_args(diagnose_parser)
    diagnose_parser.add_argument("--duration", type=float, default=10.0)

    sim_ppo_parser = subparsers.add_parser("train-sim-ppo", help="Train PPO with numeric observations in DinoEnv")
    sim_ppo_parser.add_argument("--total-timesteps", type=int, default=250_000)
    sim_ppo_parser.add_argument("--seed", type=int, default=42)
    sim_ppo_parser.add_argument("--learning-rate", type=float, default=3e-4)
    sim_ppo_parser.add_argument("--n-steps", type=int, default=1024)
    sim_ppo_parser.add_argument("--batch-size", type=int, default=64)
    sim_ppo_parser.add_argument("--gamma", type=float, default=0.99)
    sim_ppo_parser.add_argument("--n-envs", type=int, default=4)
    sim_ppo_parser.add_argument("--device", type=str, default="cpu")
    sim_ppo_parser.add_argument("--output-dir", type=Path, default=DEFAULT_PPO_ARTIFACTS_DIR)
    sim_ppo_parser.add_argument("--source-model", type=Path, default=None, help="Resume sim PPO from a checkpoint")
    sim_ppo_parser.add_argument("--best", action="store_true", help="Resume sim PPO from best_sim_model.zip if available")
    _add_sim_transfer_args(sim_ppo_parser)

    eval_sim_ppo_parser = subparsers.add_parser("eval-sim-ppo", help="Evaluate a PPO model in the simulator without rendering")
    eval_sim_ppo_parser.add_argument("--model-path", type=Path, default=None)
    eval_sim_ppo_parser.add_argument("--best", action="store_true", help="Use the best available PPO model")
    eval_sim_ppo_parser.add_argument("--episodes", type=int, default=10)
    eval_sim_ppo_parser.add_argument("--seed", type=int, default=42)
    eval_sim_ppo_parser.add_argument("--device", type=str, default="cpu")
    eval_sim_ppo_parser.add_argument("--output-dir", type=Path, default=DEFAULT_PPO_ARTIFACTS_DIR)

    real_ppo_parser = subparsers.add_parser("train-real-ppo", help="Continue PPO training on the real Chrome Dino")
    _add_real_args(real_ppo_parser)
    real_ppo_parser.add_argument("--total-timesteps", type=int, default=5_000)
    real_ppo_parser.add_argument("--source-model", type=Path, default=None)
    real_ppo_parser.add_argument("--seed", type=int, default=42)
    real_ppo_parser.add_argument("--device", type=str, default="cpu")
    real_ppo_parser.add_argument("--step-interval-s", type=float, default=RealEnvConfig().step_interval_s)
    real_ppo_parser.add_argument("--reset-wait-s", type=float, default=RealEnvConfig().reset_wait_s)
    real_ppo_parser.add_argument("--output-dir", type=Path, default=DEFAULT_PPO_ARTIFACTS_DIR)

    play_sim_ppo_parser = subparsers.add_parser("play-sim-ppo", help="Run a PPO model in DinoEnv")
    play_sim_ppo_parser.add_argument("--model-path", type=Path, default=None)
    play_sim_ppo_parser.add_argument("--best", action="store_true", help="Use real_model.zip if available, otherwise sim_model.zip")
    play_sim_ppo_parser.add_argument("--duration", type=float, default=30.0)
    play_sim_ppo_parser.add_argument("--output-dir", type=Path, default=DEFAULT_PPO_ARTIFACTS_DIR)
    play_sim_ppo_parser.add_argument("--device", type=str, default="cpu")
    play_sim_ppo_parser.add_argument("--no-display", action="store_true")

    play_real_ppo_parser = subparsers.add_parser("play-real-ppo", help="Run a pure PPO model on the real Chrome Dino")
    _add_real_args(play_real_ppo_parser)
    play_real_ppo_parser.add_argument("--model-path", type=Path, default=None)
    play_real_ppo_parser.add_argument("--best", action="store_true", help="Use real_model.zip if available, otherwise sim_model.zip")
    play_real_ppo_parser.add_argument("--duration", type=float, default=None)
    play_real_ppo_parser.add_argument("--output-dir", type=Path, default=DEFAULT_PPO_ARTIFACTS_DIR)
    play_real_ppo_parser.add_argument("--device", type=str, default="cpu")

    evolve_parser = subparsers.add_parser("evolve-real", help="Legacy: evolve NEAT genomes directly on Chrome Dino")
    _add_real_args(evolve_parser)
    evolve_parser.add_argument("--generations", type=int, default=NeatConfig().generations)
    evolve_parser.add_argument("--population", type=int, default=NeatConfig().population)
    evolve_parser.add_argument("--output-dir", type=Path, default=NeatConfig().output_dir)
    evolve_parser.add_argument("--resume", action="store_true", help="Resume from the latest NEAT checkpoint in output-dir")
    evolve_parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Resume from a specific neat-checkpoint-* file")

    gym_train_parser = subparsers.add_parser(
        "train-sim-parallel",
        help="Legacy: train a simulator policy with Gymnasium in parallel",
    )
    gym_train_parser.add_argument("--generations", type=int, default=GymParallelConfig().generations)
    gym_train_parser.add_argument("--population", type=int, default=GymParallelConfig().population)
    gym_train_parser.add_argument("--workers", type=int, default=GymParallelConfig().workers)
    gym_train_parser.add_argument("--rollout-steps", type=int, default=GymParallelConfig().rollout_steps)
    gym_train_parser.add_argument("--action-latency-steps", type=int, default=1)
    gym_train_parser.add_argument("--observation-latency-steps", type=int, default=1)
    gym_train_parser.add_argument("--observation-noise", type=float, default=0.03)
    gym_train_parser.add_argument("--domain-randomization-strength", type=float, default=0.18)
    gym_train_parser.add_argument("--output-dir", type=Path, default=GymParallelConfig().output_dir)

    gym_play_parser = subparsers.add_parser(
        "play-sim-parallel",
        help="Legacy: visualize a Gymnasium-trained simulator policy",
    )
    gym_play_parser.add_argument("--model-path", type=Path, default=None)
    gym_play_parser.add_argument("--best", action="store_true", help="Use artifacts/gym_parallel/best_policy.npz")
    gym_play_parser.add_argument("--duration", type=float, default=30.0)
    gym_play_parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "gym_parallel")

    play_neat_parser = subparsers.add_parser("play-real-neat", help="Legacy: run the best NEAT genome on Chrome Dino")
    _add_real_args(play_neat_parser)
    play_neat_parser.add_argument("--model-path", type=Path, default=None)
    play_neat_parser.add_argument("--best", action="store_true", help="Use artifacts/neat/winner.pkl or best_genome.pkl")
    play_neat_parser.add_argument("--duration", type=float, default=None)
    play_neat_parser.add_argument("--output-dir", type=Path, default=NeatConfig().output_dir)

    results_parser = subparsers.add_parser("results", help="Summarize PPO artifacts first, then legacy NEAT artifacts")
    results_parser.add_argument("--output-dir", type=Path, default=DEFAULT_PPO_ARTIFACTS_DIR)
    results_parser.add_argument("--neat-output-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)

    return parser


def _add_real_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--window-title", type=str, default=None)
    parser.add_argument("--frame-rate", type=float, default=RealGameConfig().frame_rate)
    parser.add_argument("--focus-refresh-s", type=float, default=RealGameConfig().focus_refresh_s)
    parser.add_argument("--manual-region", nargs=4, type=int, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))


def _add_sim_transfer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--action-latency-steps", type=int, default=1)
    parser.add_argument("--observation-latency-steps", type=int, default=1)
    parser.add_argument("--observation-noise", type=float, default=0.03)
    parser.add_argument("--domain-randomization-strength", type=float, default=0.18)


def _load_transfer_context() -> tuple[SimConfig, dict[str, object]]:
    handoff = load_gym_handoff(Path("artifacts") / "gym_parallel")
    if handoff is None:
        return SimConfig(), {}

    sim_config = SimConfig(**handoff.recommended_sim_config)
    neat_overrides: dict[str, object] = {}
    for key in ("episode_distance_px", "seed", "teacher_source"):
        value = handoff.recommended_neat_config.get(key)
        if value is not None and value != "":
            neat_overrides[key] = value
    return sim_config, neat_overrides


def _real_config_from_args(args) -> RealGameConfig:
    manual_region = tuple(args.manual_region) if args.manual_region is not None else DEFAULT_MANUAL_REGION
    candidates = ("dino", "Chrome") if args.window_title is None else (args.window_title, "dino", "Chrome")
    return RealGameConfig(
        debug=not args.no_debug,
        manual_region=manual_region,
        window_title_candidates=candidates,
        frame_rate=args.frame_rate,
        focus_refresh_s=args.focus_refresh_s,
    )


def _sim_config_from_ppo_args(args) -> SimConfig:
    return SimConfig(
        action_latency_steps=args.action_latency_steps,
        observation_latency_steps=args.observation_latency_steps,
        observation_noise=args.observation_noise,
        domain_randomization_strength=args.domain_randomization_strength,
        seed=args.seed,
    )


def _real_env_config_from_ppo_args(args, real_config: RealGameConfig) -> RealEnvConfig:
    return RealEnvConfig(
        step_interval_s=args.step_interval_s,
        reset_wait_s=args.reset_wait_s,
        show_debug=real_config.debug,
        enable_action_gate=False,
    )


def run_play_real(
    duration: float | None = None,
    debug: bool = True,
    *,
    window_title: str | None = None,
    frame_rate: float | None = None,
    focus_refresh_s: float | None = None,
    manual_region: tuple[int, int, int, int] | None = None,
):
    if manual_region is None:
        manual_region = DEFAULT_MANUAL_REGION
    candidates = ("dino", "Chrome") if window_title is None else (window_title, "dino", "Chrome")
    real_config = RealGameConfig(debug=debug, manual_region=manual_region, window_title_candidates=candidates)
    if frame_rate is not None:
        real_config.frame_rate = frame_rate
    if focus_refresh_s is not None:
        real_config.focus_refresh_s = focus_refresh_s
    runner = DinoRealGameRunner(real_config=real_config, vision_config=DinoVisionConfig())
    stats = runner.run(duration_s=duration)
    print(f"real_game: frames={stats.frames} jumps={stats.jumps} ducks={stats.ducks} noops={stats.noops}")
    return stats


def run_diagnose_real(duration: float, real_config: RealGameConfig):
    sim_config, _ = _load_transfer_context()
    runner = DinoRealGameRunner(real_config=real_config, vision_config=DinoVisionConfig())
    observation_builder = RealRLObservationBuilder(sim_config=sim_config)
    start = time.time()
    frame_interval = 1.0 / max(1.0, runner.real_config.frame_rate)

    try:
        runner.capture.resolve_region()
    except Exception as exc:
        print(f"diagnose: {exc}")
        return None

    try:
        while time.time() - start < duration:
            now = time.time()
            frame = runner.capture.capture()
            state = runner.analyzer.analyze(frame, timestamp=now)
            obs = observation_builder.build(state)
            nearest = state.nearest_obstacle
            obstacle_text = (
                f"{nearest.label}@{nearest.rect.x},{nearest.rect.y},{nearest.rect.w},{nearest.rect.h}"
                if nearest is not None
                else "none"
            )
            print(
                "diagnose: speed={speed:.1f} dist={dist} grounded={grounded} nearest={nearest} obs={obs}".format(
                    speed=state.estimated_speed_px_s,
                    dist=f"{state.nearest_distance_px:.1f}" if state.nearest_distance_px is not None else "none",
                    grounded=state.player_box.bottom >= state.ground_y - 4,
                    nearest=obstacle_text,
                    obs=",".join(f"{value:.3f}" for value in obs[:6]),
                )
            )
            if real_config.debug:
                import cv2

                cv2.imshow("DinoIA - Diagnose", runner.analyzer.draw_overlay(state))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            time.sleep(max(0.0, frame_interval - (time.time() - now)))
    finally:
        if real_config.debug:
            import cv2

            cv2.destroyAllWindows()


def run_evolve_real(args) -> Path:
    sim_config, neat_overrides = _load_transfer_context()
    neat_kwargs = {
        "generations": args.generations,
        "population": args.population,
        "output_dir": args.output_dir,
    }
    neat_kwargs.update(neat_overrides)
    neat_kwargs["generations"] = args.generations
    neat_kwargs["population"] = args.population
    neat_kwargs["output_dir"] = args.output_dir
    neat_config = NeatConfig(**neat_kwargs)
    resume_checkpoint = _resolve_resume_checkpoint(args)
    winner_path = evolve_real_neat(
        neat_config=neat_config,
        real_config=_real_config_from_args(args),
        vision_config=DinoVisionConfig(),
        sim_config=sim_config,
        resume_checkpoint=resume_checkpoint,
    )
    print(f"evolve-real: winner={winner_path}")
    return winner_path


def run_train_sim_ppo(args):
    source_model_path = args.source_model
    if args.best and source_model_path is None:
        source_model_path = Path(args.output_dir) / "best_sim_model.zip"
    result = train_sim_ppo_agent(
        total_timesteps=args.total_timesteps,
        output_dir=args.output_dir,
        source_model_path=source_model_path,
        sim_config=_sim_config_from_ppo_args(args),
        seed=args.seed,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        device=args.device,
    )
    print(
        "train-sim-ppo: source={source} model={model} best={best} config={config} history={history}".format(
            source=result.source_model_path or "none",
            model=result.model_path,
            best=result.best_model_path or "none",
            config=result.config_path,
            history=result.history_path,
        )
    )
    return result


def run_train_real_ppo(args):
    real_config = _real_config_from_args(args)
    result = train_real_ppo_agent(
        total_timesteps=args.total_timesteps,
        output_dir=args.output_dir,
        source_model_path=args.source_model,
        real_config=real_config,
        vision_config=DinoVisionConfig(),
        env_config=_real_env_config_from_ppo_args(args, real_config),
        seed=args.seed,
        device=args.device,
    )
    print(
        "train-real-ppo: source={source} model={model} best={best} config={config} history={history}".format(
            source=result.source_model_path,
            model=result.model_path,
            best=result.best_model_path or "none",
            config=result.config_path,
            history=result.history_path,
        )
    )
    return result


def run_eval_sim_ppo(args):
    model_path = None if args.best or args.model_path is None else args.model_path
    result = evaluate_sim_ppo_agent(
        model_path=model_path,
        output_dir=args.output_dir,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
    )
    print(
        "eval-sim-ppo: episodes={episodes} mean_reward={mean_reward:.3f} "
        "passed={passed:.3f} max_passed={max_passed} actions={actions} model={model}".format(
            episodes=result.episodes,
            mean_reward=result.mean_reward,
            passed=result.mean_passed_obstacles,
            max_passed=result.max_passed_obstacles,
            actions=result.action_counts,
            model=result.model_path,
        )
    )
    return result


def run_play_sim_ppo(args):
    model_path = None if args.best or args.model_path is None else args.model_path
    result = play_sim_ppo_agent(
        model_path=model_path,
        output_dir=args.output_dir,
        duration_s=args.duration,
        display=not args.no_display,
        device=args.device,
    )
    print(
        "play-sim-ppo: steps={steps} reward={reward:.3f} passed={passed} episodes={episodes} model={model}".format(
            steps=result.steps,
            reward=result.reward,
            passed=result.passed_obstacles,
            episodes=result.episodes,
            model=result.model_path,
        )
    )
    return result


def run_play_real_ppo(args):
    model_path = None if args.best or args.model_path is None else args.model_path
    result = play_real_ppo_agent(
        model_path=model_path,
        output_dir=args.output_dir,
        duration_s=args.duration,
        real_config=_real_config_from_args(args),
        vision_config=DinoVisionConfig(),
        device=args.device,
    )
    print(
        "play-real-ppo: steps={steps} reward={reward:.3f} passed={passed} episodes={episodes} "
        "jumps={jumps} ducks={ducks} noops={noops} model={model}".format(
            steps=result.steps,
            reward=result.reward,
            passed=result.passed_obstacles,
            episodes=result.episodes,
            jumps=result.jumps,
            ducks=result.ducks,
            noops=result.noops,
            model=result.model_path,
        )
    )
    return result


def _resolve_resume_checkpoint(args) -> Path | None:
    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_checkpoint}")
        return args.resume_checkpoint
    if not args.resume:
        return None
    checkpoints = list_neat_checkpoints(args.output_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No NEAT checkpoint found in {args.output_dir}. Run evolve-real without --resume first.")
    return checkpoints[0]


def run_play_real_neat(args):
    sim_config, neat_overrides = _load_transfer_context()
    model_path = args.model_path
    if model_path is None:
        model_path = select_best_model(args.output_dir)
    if model_path is None:
        raise FileNotFoundError("No NEAT genome was found in artifacts/neat. Run evolve-real first.")
    neat_kwargs = {
        "output_dir": args.output_dir,
    }
    neat_kwargs.update(neat_overrides)
    neat_kwargs["output_dir"] = args.output_dir
    runner = DinoRealNeatRunner(
        model_path=model_path,
        real_config=_real_config_from_args(args),
        vision_config=DinoVisionConfig(),
        sim_config=sim_config,
        neat_config=NeatConfig(**neat_kwargs),
    )
    result = runner.run(duration_s=args.duration)
    print(
        f"real_neat: steps={result.steps} jumps={result.jumps} ducks={result.ducks} "
        f"noops={result.noops} safety_overrides={result.safety_overrides} model={result.model_path}"
    )
    return result


def run_doctor(output_dir: Path | None = None):
    report = run_environment_doctor(output_dir)
    print(format_doctor_report(report))
    if report.ok:
        print("doctor: ready for PPO")
        print("doctor: next -> python -m dinoia train-sim-ppo --total-timesteps 250000")
    else:
        print("doctor: fix the blocked checks before training PPO")
    return report


def run_results(output_dir: Path | None = None, neat_output_dir: Path | None = None):
    ppo_summary = summarize_ppo(output_dir)
    neat_summary = summarize_project(neat_output_dir)
    gym_handoff = load_gym_handoff(Path("artifacts") / "gym_parallel")
    print(f"results: ppo_output_dir={ppo_summary.output_dir}")
    print(f"results: ppo_best_model={ppo_summary.best_model_path or 'none'}")
    print(
        "results: ppo_best_eval_mean_reward={value}".format(
            value=ppo_summary.best_eval_mean_reward if ppo_summary.best_eval_mean_reward is not None else "none"
        )
    )
    print(
        "results: ppo_latest_eval_mean_reward={value}".format(
            value=ppo_summary.latest_eval_mean_reward if ppo_summary.latest_eval_mean_reward is not None else "none"
        )
    )
    print(f"results: ppo_sim_model={ppo_summary.sim_model_path or 'none'}")
    print(f"results: ppo_best_sim_model={ppo_summary.best_sim_model_path or 'none'}")
    print(f"results: ppo_sim_vecnormalize={ppo_summary.sim_vecnormalize_path or 'none'}")
    print(f"results: ppo_best_sim_vecnormalize={ppo_summary.best_sim_vecnormalize_path or 'none'}")
    print(f"results: ppo_real_model={ppo_summary.real_model_path or 'none'}")
    print(f"results: ppo_best_real_model={ppo_summary.best_real_model_path or 'none'}")
    print(f"results: ppo_real_vecnormalize={ppo_summary.real_vecnormalize_path or 'none'}")
    print(f"results: ppo_best_real_vecnormalize={ppo_summary.best_real_vecnormalize_path or 'none'}")
    print(f"results: ppo_best_model_vecnormalize={ppo_summary.best_model_vecnormalize_path or 'none'}")
    print(f"results: ppo_history={ppo_summary.history_path or 'none'}")
    print(f"results: ppo_eval_history={ppo_summary.evaluation_history_path or 'none'}")
    print(f"results: ppo_latest_summary={ppo_summary.summary_path or 'none'}")
    print(f"results: ppo_latest_eval={ppo_summary.latest_eval_path or 'none'}")
    print(f"results: ppo_best_sim_eval={ppo_summary.best_sim_eval_path or 'none'}")
    print(f"results: ppo_best_real_eval={ppo_summary.best_real_eval_path or 'none'}")
    print(f"results: ppo_latest_stage={ppo_summary.latest_stage or 'none'}")
    print(
        "results: ppo_latest_total_timesteps={value}".format(
            value=ppo_summary.latest_total_timesteps if ppo_summary.latest_total_timesteps is not None else "none"
        )
    )
    print(f"results: legacy_neat_output_dir={neat_summary.output_dir}")
    print(f"results: legacy_neat_winner={neat_summary.winner_path or 'none'}")
    print(f"results: legacy_neat_best_genome={neat_summary.best_genome_path or 'none'}")
    print(f"results: legacy_neat_latest_checkpoint={neat_summary.latest_checkpoint_path or 'none'}")
    print(f"results: legacy_neat_history={neat_summary.history_path or 'none'}")
    print(
        "results: legacy_neat_best_fitness={value}".format(
            value=neat_summary.best_fitness if neat_summary.best_fitness is not None else "none"
        )
    )
    print(f"results: legacy_gym_handoff={gym_handoff.source_policy_path if gym_handoff is not None else 'none'}")
    if ppo_summary.best_model_path is None:
        print("results: next -> python -m dinoia train-sim-ppo --total-timesteps 250000")
    elif ppo_summary.real_model_path is None:
        print("results: next -> python -m dinoia train-real-ppo --total-timesteps 5000")
    else:
        print("results: next -> python -m dinoia play-real-ppo --best")
    return ppo_summary


def run_train_sim_parallel(args) -> Path:
    sim_config = SimConfig(
        action_latency_steps=args.action_latency_steps,
        observation_latency_steps=args.observation_latency_steps,
        observation_noise=args.observation_noise,
        domain_randomization_strength=args.domain_randomization_strength,
    )
    config = GymParallelConfig(
        generations=args.generations,
        population=args.population,
        workers=args.workers,
        rollout_steps=args.rollout_steps,
        output_dir=args.output_dir,
        sim_config=sim_config,
    )
    policy_path = train_parallel_gymnasium(config=config)
    print(f"train-sim-parallel: policy={policy_path}")
    return policy_path


def run_play_sim_parallel(args):
    model_path = args.model_path
    if args.best or model_path is None:
        model_path = None
    result = visualize_parallel_policy(
        model_path=model_path,
        output_dir=args.output_dir,
        duration_s=args.duration,
    )
    print(
        "play-sim-parallel: frames={frames} reward={reward:.3f} passed={passed} episodes={episodes} model={model}".format(
            frames=result.frames,
            reward=result.reward,
            passed=result.passed_obstacles,
            episodes=result.episodes,
            model=result.model_path,
        )
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command is None:
            run_doctor()
            return 0
        if args.command == "doctor":
            run_doctor(args.output_dir)
            return 0
        if args.command == "play-real":
            real_config = _real_config_from_args(args)
            run_play_real(
                debug=real_config.debug,
                window_title=args.window_title,
                frame_rate=args.frame_rate,
                focus_refresh_s=args.focus_refresh_s,
                manual_region=real_config.manual_region,
            )
            return 0
        if args.command == "diagnose-real":
            run_diagnose_real(args.duration, _real_config_from_args(args))
            return 0
        if args.command == "evolve-real":
            run_evolve_real(args)
            return 0
        if args.command == "train-sim-ppo":
            run_train_sim_ppo(args)
            return 0
        if args.command == "eval-sim-ppo":
            run_eval_sim_ppo(args)
            return 0
        if args.command == "train-real-ppo":
            run_train_real_ppo(args)
            return 0
        if args.command == "play-sim-ppo":
            run_play_sim_ppo(args)
            return 0
        if args.command == "play-real-ppo":
            run_play_real_ppo(args)
            return 0
        if args.command == "train-sim-parallel":
            run_train_sim_parallel(args)
            return 0
        if args.command == "play-sim-parallel":
            run_play_sim_parallel(args)
            return 0
        if args.command == "play-real-neat":
            run_play_real_neat(args)
            return 0
        if args.command == "results":
            run_results(args.output_dir, args.neat_output_dir)
            return 0
        parser.print_help()
        return 1
    except KeyboardInterrupt:
        print("dinoia: interrupted")
        return 130
