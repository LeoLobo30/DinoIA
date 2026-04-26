from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config import DinoVisionConfig, RealGameConfig, SimConfig, TrainingConfig
from .diagnostics import format_doctor_report, run_doctor as run_environment_doctor
from .project_state import DEFAULT_ARTIFACTS_DIR, select_best_model, select_latest_model, summarize_project
from .real_game import DinoRealGameRunner
from .real_rl import RealRLObservationBuilder

DEFAULT_MANUAL_REGION = (1300, 70, 1200, 400)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dinoia", description="DinoIA portfolio project")
    subparsers = parser.add_subparsers(dest="command")

    doctor_parser = subparsers.add_parser("doctor", help="Check whether the training environment is ready")
    doctor_parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)

    play_parser = subparsers.add_parser("play-real", help="Run the Chrome Dino vision agent")
    play_parser.add_argument("--duration", type=float, default=None)
    play_parser.add_argument("--no-debug", action="store_true")
    play_parser.add_argument("--window-title", type=str, default=None)
    play_parser.add_argument("--frame-rate", type=float, default=RealGameConfig().frame_rate)
    play_parser.add_argument("--focus-refresh-s", type=float, default=RealGameConfig().focus_refresh_s)
    play_parser.add_argument("--manual-region", nargs=4, type=int, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))

    diagnose_parser = subparsers.add_parser("diagnose-real", help="Inspect the real-game capture and RL observation")
    diagnose_parser.add_argument("--duration", type=float, default=10.0)
    diagnose_parser.add_argument("--no-debug", action="store_true")
    diagnose_parser.add_argument("--window-title", type=str, default=None)
    diagnose_parser.add_argument("--frame-rate", type=float, default=RealGameConfig().frame_rate)
    diagnose_parser.add_argument("--focus-refresh-s", type=float, default=RealGameConfig().focus_refresh_s)
    diagnose_parser.add_argument("--manual-region", nargs=4, type=int, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))

    train_parser = subparsers.add_parser("train", help="Train the Gymnasium DQN agent")
    train_parser.add_argument("--preset", choices=("quick", "standard", "long", "real-short"), default="standard")
    train_parser.add_argument("--timesteps", type=int, default=None, help="Override preset timesteps")
    train_parser.add_argument("--output-dir", type=Path, default=TrainingConfig().output_dir)
    train_parser.add_argument("--n-envs", type=int, default=None, help="Number of parallel training environments")
    train_parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    train_parser.add_argument("--resume", type=str, default="auto", help="auto, none, or a checkpoint/model path")
    train_parser.add_argument("--fresh", action="store_true", help="Ignore old checkpoints and start from scratch")

    train_real_parser = subparsers.add_parser("train-real", help="Train DQN directly on the real Chrome Dino")
    train_real_parser.add_argument("--timesteps", type=int, default=5_000)
    train_real_parser.add_argument("--output-dir", type=Path, default=TrainingConfig().output_dir)
    train_real_parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    train_real_parser.add_argument("--resume", type=str, default="auto", help="auto, none, or a checkpoint/model path")
    train_real_parser.add_argument("--fresh", action="store_true", help="Ignore old checkpoints and start from scratch")
    train_real_parser.add_argument("--no-debug", action="store_true")
    train_real_parser.add_argument("--window-title", type=str, default=None)
    train_real_parser.add_argument("--frame-rate", type=float, default=RealGameConfig().frame_rate)
    train_real_parser.add_argument("--focus-refresh-s", type=float, default=RealGameConfig().focus_refresh_s)
    train_real_parser.add_argument("--manual-region", nargs=4, type=int, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))
    train_real_parser.add_argument("--reset-wait-s", type=float, default=0.8)

    combo_parser = subparsers.add_parser("play-and-train", help="Run the real agent and train DQN in parallel")
    combo_parser.add_argument("--duration", type=float, default=None)
    combo_parser.add_argument("--preset", choices=("quick", "standard", "long", "real-short"), default="standard")
    combo_parser.add_argument("--timesteps", type=int, default=None, help="Override preset timesteps")
    combo_parser.add_argument("--output-dir", type=Path, default=TrainingConfig().output_dir)
    combo_parser.add_argument("--n-envs", type=int, default=None, help="Number of parallel training environments")
    combo_parser.add_argument("--device", type=str, default="auto")
    combo_parser.add_argument("--no-debug", action="store_true", help="Deprecated alias for headless mode")
    combo_parser.add_argument("--show-capture", action="store_true", help="Show the real-game capture window")
    combo_parser.add_argument("--window-title", type=str, default=None)
    combo_parser.add_argument("--frame-rate", type=float, default=12.0)
    combo_parser.add_argument("--focus-refresh-s", type=float, default=10.0)
    combo_parser.add_argument("--manual-region", nargs=4, type=int, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))

    real_rl_parser = subparsers.add_parser("play-real-rl", help="Run the trained DQN on the real Chrome Dino")
    real_rl_group = real_rl_parser.add_mutually_exclusive_group()
    real_rl_group.add_argument("--model-path", type=Path, default=None)
    real_rl_group.add_argument("--latest", action="store_true", help="Use the latest available model")
    real_rl_group.add_argument("--best", action="store_true", help="Use the best available model")
    real_rl_parser.add_argument("--duration", type=float, default=None)
    real_rl_parser.add_argument("--device", type=str, default="auto")
    real_rl_parser.add_argument("--no-debug", action="store_true")
    real_rl_parser.add_argument("--window-title", type=str, default=None)
    real_rl_parser.add_argument("--frame-rate", type=float, default=RealGameConfig().frame_rate)
    real_rl_parser.add_argument("--focus-refresh-s", type=float, default=RealGameConfig().focus_refresh_s)
    real_rl_parser.add_argument("--manual-region", nargs=4, type=int, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained DQN model")
    eval_group = eval_parser.add_mutually_exclusive_group()
    eval_group.add_argument("--model-path", type=Path, default=None)
    eval_group.add_argument("--latest", action="store_true")
    eval_group.add_argument("--best", action="store_true")
    eval_parser.add_argument("--episodes", type=int, default=5)
    eval_parser.add_argument("--render", action="store_true")
    eval_parser.add_argument("--seed", type=int, default=42)

    results_parser = subparsers.add_parser("results", help="Summarize the last training run")
    results_parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)

    benchmark_parser = subparsers.add_parser("benchmark", help="Compare CPU and GPU training throughput")
    benchmark_parser.add_argument("--timesteps", type=int, default=3_000)
    benchmark_parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))

    return parser


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
    runner = DinoRealGameRunner(
        real_config=real_config,
        vision_config=DinoVisionConfig(),
    )
    stats = runner.run(duration_s=duration)
    print(
        f"real_game: frames={stats.frames} jumps={stats.jumps} ducks={stats.ducks} noops={stats.noops}"
    )
    return stats


def run_diagnose_real(
    duration: float = 10.0,
    debug: bool = False,
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
    observation_builder = RealRLObservationBuilder()
    start = time.time()
    frame_interval = 1.0 / max(1.0, runner.real_config.frame_rate)
    last_focus_at = 0.0

    if runner.real_config.focus_window:
        try:
            runner.capture.focus_window()
            last_focus_at = time.time()
        except Exception:
            pass

    try:
        runner.capture.resolve_region()
    except Exception as exc:
        print(f"diagnose: {exc}")
        return None

    try:
        while True:
            now = time.time()
            if now - start >= duration:
                break

            if (
                runner.real_config.focus_window
                and runner.real_config.keep_window_foreground
                and now - last_focus_at >= runner.real_config.focus_refresh_s
            ):
                try:
                    runner.capture.focus_window()
                    last_focus_at = now
                except Exception:
                    pass

            try:
                frame = runner.capture.capture()
            except Exception as exc:
                print(f"diagnose: {exc}")
                break
            state = runner.analyzer.analyze(frame, timestamp=now)
            obs = observation_builder.build(state)
            player = state.player_box
            nearest = state.nearest_obstacle
            obstacle_text = (
                f"{nearest.label}@{nearest.rect.x},{nearest.rect.y},{nearest.rect.w},{nearest.rect.h}"
                if nearest is not None
                else "none"
            )
            print(
                "diagnose: speed={speed:.1f} dist={dist} grounded={grounded} ducking={ducking} "
                "player={player} nearest={nearest} count={count} obs={obs}".format(
                    speed=state.estimated_speed_px_s,
                    dist=f"{state.nearest_distance_px:.1f}" if state.nearest_distance_px is not None else "none",
                    grounded=state.player_box.bottom >= state.ground_y - 4,
                    ducking=bool(obs[3] >= 0.5),
                    player=f"{player.x},{player.y},{player.w},{player.h}",
                    nearest=obstacle_text,
                    count=len(state.obstacles),
                    obs=",".join(f"{value:.3f}" for value in obs[:6]),
                )
            )

            if debug:
                overlay = runner.analyzer.draw_overlay(state)
                import cv2

                cv2.imshow("DinoIA - Diagnose", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            sleep_for = max(0.0, frame_interval - (time.time() - now))
            time.sleep(sleep_for)
    finally:
        if debug:
            import cv2

            cv2.destroyAllWindows()


def run_doctor(output_dir: Path | None = None):
    report = run_environment_doctor(output_dir)
    print(format_doctor_report(report))
    if report.ok:
        print("doctor: ready to train")
        print("doctor: next -> python -m dinoia train --preset quick --device auto --resume auto")
    else:
        print("doctor: fix the blocked checks before a long training run")
    return report


def run_train(
    *,
    preset: str = "standard",
    timesteps: int | None = None,
    output_dir: Path | None = None,
    n_envs: int | None = None,
    device: str = "auto",
    resume: str | Path = "auto",
    fresh: bool = False,
):
    output_dir = output_dir or TrainingConfig().output_dir
    from .rl.train import train_dqn

    training_config = TrainingConfig(
        preset=preset,
        output_dir=output_dir,
        n_envs=n_envs if n_envs is not None else TrainingConfig().n_envs,
        device=device,
        resume=str(resume),
        fresh=fresh,
    )
    if timesteps is not None:
        training_config.total_timesteps = timesteps
    model_path = train_dqn(
        training_config=training_config,
        preset=preset,
        total_timesteps=timesteps,
        output_dir=output_dir,
        n_envs=n_envs,
        device=device,
        resume=resume,
        fresh=fresh,
    )
    print(f"train: saved_model={model_path}")
    return model_path


def run_train_real(
    *,
    timesteps: int = 5_000,
    output_dir: Path | None = None,
    device: str = "auto",
    resume: str = "auto",
    fresh: bool = False,
    debug: bool = False,
    window_title: str | None = None,
    frame_rate: float | None = None,
    focus_refresh_s: float | None = None,
    manual_region: tuple[int, int, int, int] | None = None,
    reset_wait_s: float = 0.8,
):
    if manual_region is None:
        manual_region = DEFAULT_MANUAL_REGION
    candidates = ("dino", "Chrome") if window_title is None else (window_title, "dino", "Chrome")
    real_config = RealGameConfig(debug=debug, manual_region=manual_region, window_title_candidates=candidates)
    if frame_rate is not None:
        real_config.frame_rate = frame_rate
    if focus_refresh_s is not None:
        real_config.focus_refresh_s = focus_refresh_s

    from .real_env import RealEnvConfig
    from .rl.train_real import train_real_dqn

    model_path = train_real_dqn(
        total_timesteps=timesteps,
        output_dir=output_dir or TrainingConfig().output_dir,
        device=device,
        resume=resume,
        fresh=fresh,
        real_config=real_config,
        vision_config=DinoVisionConfig(),
        env_config=RealEnvConfig(
            step_interval_s=1.0 / max(1.0, real_config.frame_rate),
            reset_wait_s=reset_wait_s,
            show_debug=debug,
        ),
    )
    print(f"train-real: saved_model={model_path}")
    return model_path


def run_results(output_dir: Path | None = None):
    summary = summarize_project(output_dir)
    print(f"results: output_dir={summary.output_dir}")
    if summary.training_config_path is not None:
        print(f"results: training_config={summary.training_config_path}")
    if summary.model_path is not None:
        print(f"results: latest_model={summary.model_path}")
    if summary.best_model_path is not None:
        print(f"results: best_model={summary.best_model_path}")
    if summary.latest_checkpoint_path is not None:
        print(f"results: latest_checkpoint={summary.latest_checkpoint_path}")
    if summary.evaluation is not None:
        evaluation = summary.evaluation
        print(
            "results: eval_mean_reward={mean_reward:.2f} eval_mean_length={mean_length:.1f} episodes={episodes}".format(
                mean_reward=evaluation["mean_reward"],
                mean_length=evaluation["mean_length"],
                episodes=len(evaluation["rewards"]),
            )
        )
    else:
        print("results: evaluation=none")

    if summary.training_config is not None:
        training = summary.training_config.get("training", {})
        resume_info = summary.training_config.get("resume", {})
        preset = training.get("preset", "standard")
        total_timesteps = training.get("total_timesteps", "unknown")
        device = training.get("device", "auto")
        print(f"results: preset={preset} timesteps={total_timesteps} device={device}")
        if resume_info:
            print(
                f"results: resume_requested={resume_info.get('requested')} resumed_from={resume_info.get('resumed_from')} fresh={resume_info.get('fresh')}"
            )
    if summary.checkpoints:
        print(f"results: checkpoints={len(summary.checkpoints)}")
    else:
        print("results: checkpoints=0")
    if summary.model_path is None:
        print("results: next -> python -m dinoia train --preset standard --device auto --resume auto")
    else:
        print("results: next -> python -m dinoia evaluate --latest")
    return summary


def run_evaluate(
    model_path: Path | None,
    episodes: int,
    render: bool,
    seed: int,
    *,
    latest: bool = False,
    best: bool = False,
):
    from .rl.evaluate import evaluate_model

    summary = evaluate_model(model_path, episodes=episodes, render=render, seed=seed, latest=latest, best=best)
    print(
        "evaluate: model={model} mean_reward={mean_reward:.2f} std_reward={std_reward:.2f} mean_length={mean_length:.1f}".format(
            model=summary["model_path"],
            mean_reward=summary["mean_reward"],
            std_reward=summary["std_reward"],
            mean_length=summary["mean_length"],
        )
    )
    return summary


def run_benchmark(timesteps: int, device: str):
    from .diagnostics import benchmark_training

    devices = [device] if device != "auto" else None
    results = benchmark_training(timesteps=timesteps, devices=devices)
    print(f"benchmark: timesteps={timesteps}")
    for result in results:
        if "error" in result:
            print(f"benchmark: device={result['device']} error={result['error']}")
            continue
        print(
            "benchmark: device={device} elapsed_s={elapsed:.2f} timesteps_per_second={tps:.1f}".format(
                device=result["device"],
                elapsed=result["elapsed_s"],
                tps=result["timesteps_per_second"],
            )
        )
    valid = [result for result in results if "timesteps_per_second" in result]
    if valid:
        fastest = max(valid, key=lambda item: item["timesteps_per_second"])
        print(f"benchmark: recommended_device={fastest['device']}")
    return results


def run_play_real_rl(
    model_path: Path,
    duration: float | None = None,
    debug: bool = True,
    *,
    device: str = "auto",
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
    from .real_rl import DinoRealRLRunner

    runner = DinoRealRLRunner(
        model_path=model_path,
        real_config=real_config,
        vision_config=DinoVisionConfig(),
        device=device,
    )
    result = runner.run(duration_s=duration)
    print(f"real_rl: frames={result.stats.frames} jumps={result.stats.jumps} ducks={result.stats.ducks} noops={result.stats.noops}")
    return result


def run_play_and_train(
    duration: float | None,
    timesteps: int | None,
    output_dir: Path,
    *,
    preset: str,
    n_envs: int | None,
    device: str,
    debug: bool,
    window_title: str | None = None,
    frame_rate: float | None = None,
    focus_refresh_s: float | None = None,
    manual_region: tuple[int, int, int, int] | None = None,
):
    if manual_region is None:
        manual_region = DEFAULT_MANUAL_REGION
    results: dict[str, object] = {}
    stop_event = threading.Event()
    play_process: subprocess.Popen | None = None

    def play_worker():
        nonlocal play_process
        command = [sys.executable, "-m", "dinoia", "play-real"]
        if duration is not None:
            command.extend(["--duration", str(duration)])
        if not debug:
            command.append("--no-debug")
        if window_title is not None:
            command.extend(["--window-title", window_title])
        if frame_rate is not None:
            command.extend(["--frame-rate", str(frame_rate)])
        if focus_refresh_s is not None:
            command.extend(["--focus-refresh-s", str(focus_refresh_s)])
        if manual_region is not None:
            command.extend(["--manual-region", *[str(value) for value in manual_region]])

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        play_process = subprocess.Popen(command, creationflags=creationflags)
        results["real"] = {"pid": play_process.pid, "command": command}

    def train_worker():
        from .rl.train import train_dqn

        training_config = TrainingConfig(preset=preset, total_timesteps=timesteps or TrainingConfig().total_timesteps, output_dir=output_dir, device=device, n_envs=n_envs if n_envs is not None else TrainingConfig().n_envs)
        results["model"] = train_dqn(training_config=training_config, stop_event=stop_event, device=device)

    play_thread = threading.Thread(target=play_worker, name="dinoia-real", daemon=False)
    train_thread = threading.Thread(target=train_worker, name="dinoia-train", daemon=False)

    play_thread.start()
    train_thread.start()

    try:
        while play_thread.is_alive() or train_thread.is_alive():
            play_thread.join(timeout=0.5)
            train_thread.join(timeout=0.5)
    except KeyboardInterrupt:
        stop_event.set()
        if play_process is not None and play_process.poll() is None:
            try:
                play_process.terminate()
            except Exception:
                pass
        play_thread.join(timeout=3.0)
        train_thread.join(timeout=3.0)
        print("play-and-train: interrupted, shutting down cleanly")
    finally:
        if play_process is not None and play_process.poll() is None:
            try:
                play_process.wait(timeout=5.0)
            except Exception:
                try:
                    play_process.kill()
                except Exception:
                    pass

    return results


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
            manual_region = tuple(args.manual_region) if args.manual_region is not None else None
            run_play_real(
                duration=args.duration,
                debug=not args.no_debug,
                window_title=args.window_title,
                frame_rate=args.frame_rate,
                focus_refresh_s=args.focus_refresh_s,
                manual_region=manual_region,
            )
            return 0

        if args.command == "diagnose-real":
            manual_region = tuple(args.manual_region) if args.manual_region is not None else None
            run_diagnose_real(
                duration=args.duration,
                debug=not args.no_debug,
                window_title=args.window_title,
                frame_rate=args.frame_rate,
                focus_refresh_s=args.focus_refresh_s,
                manual_region=manual_region,
            )
            return 0

        if args.command == "train":
            run_train(
                preset=args.preset,
                timesteps=args.timesteps,
                output_dir=args.output_dir,
                n_envs=args.n_envs,
                device=args.device,
                resume=args.resume,
                fresh=args.fresh,
            )
            return 0

        if args.command == "train-real":
            manual_region = tuple(args.manual_region) if args.manual_region is not None else None
            run_train_real(
                timesteps=args.timesteps,
                output_dir=args.output_dir,
                device=args.device,
                resume=args.resume,
                fresh=args.fresh,
                debug=not args.no_debug,
                window_title=args.window_title,
                frame_rate=args.frame_rate,
                focus_refresh_s=args.focus_refresh_s,
                manual_region=manual_region,
                reset_wait_s=args.reset_wait_s,
            )
            return 0

        if args.command == "play-and-train":
            manual_region = tuple(args.manual_region) if args.manual_region is not None else None
            run_play_and_train(
                duration=args.duration,
                timesteps=args.timesteps,
                output_dir=args.output_dir,
                preset=args.preset,
                n_envs=args.n_envs,
                device=args.device,
                debug=args.show_capture and not args.no_debug,
                window_title=args.window_title,
                frame_rate=args.frame_rate,
                focus_refresh_s=args.focus_refresh_s,
                manual_region=manual_region,
            )
            return 0

        if args.command == "play-real-rl":
            manual_region = tuple(args.manual_region) if args.manual_region is not None else None
            model_path = args.model_path
            if model_path is None:
                if args.latest:
                    model_path = select_latest_model()
                else:
                    model_path = select_best_model()
            if model_path is None:
                raise FileNotFoundError("No trained model was found in artifacts/dqn. Run train first.")
            run_play_real_rl(
                model_path=model_path,
                duration=args.duration,
                debug=not args.no_debug,
                device=args.device,
                window_title=args.window_title,
                frame_rate=args.frame_rate,
                focus_refresh_s=args.focus_refresh_s,
                manual_region=manual_region,
            )
            return 0

        if args.command == "evaluate":
            model_path = args.model_path
            if model_path is None:
                model_path = select_best_model() if args.best else select_latest_model()
            if model_path is None:
                raise FileNotFoundError("No trained model was found in artifacts/dqn. Run train first.")
            run_evaluate(model_path, args.episodes, args.render, args.seed, latest=args.latest, best=args.best)
            return 0

        if args.command == "results":
            run_results(args.output_dir)
            return 0

        if args.command == "benchmark":
            run_benchmark(args.timesteps, args.device)
            return 0

        parser.print_help()
        return 1
    except KeyboardInterrupt:
        print("dinoia: interrupted")
        return 130







