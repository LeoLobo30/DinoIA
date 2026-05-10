from __future__ import annotations

import json
import pickle
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2

from .capture import DinoScreenCapture
from .config import DinoVisionConfig, NeatConfig, RealGameConfig, SimConfig
from .control import KeyboardController
from .decision import DinoRulePolicy
from .observations import OBSERVATION_SIZE
from .project_state import DEFAULT_ARTIFACTS_DIR, select_best_model
from .gym_parallel import LinearPolicy, load_best_gym_policy
from .real_rl import RealActionGate, RealRLObservationBuilder
from .types import PolicyAction, VisionState
from .vision import DinoVisionAnalyzer


@dataclass(slots=True)
class NeatEpisodeResult:
    genome_id: int
    generation: int
    fitness: float
    alive_seconds: float
    distance_px: float
    frames: int
    passed_obstacles: int
    jumps: int
    ducks: int
    noops: int
    collision: bool
    terminal_reason: str


@dataclass(slots=True)
class NeatRunResult:
    model_path: str
    steps: int
    jumps: int
    ducks: int
    noops: int
    safety_overrides: int


@dataclass(slots=True)
class EpisodeResetResult:
    ready: bool
    attempts: int
    waited_seconds: float
    reason: str


@dataclass(slots=True)
class PendingPassState:
    started_at: float
    label: str
    rect_x: int
    rect_right: int
    speed_px_s: float
    player_x: int
    airborne_seen: bool = False
    ducking_seen: bool = False


def action_from_outputs(outputs: Sequence[float]) -> PolicyAction:
    if len(outputs) != 3:
        raise ValueError(f"NEAT network must return 3 outputs, got {len(outputs)}")
    action_index = max(range(3), key=lambda index: float(outputs[index]))
    if action_index == 1:
        return PolicyAction.JUMP
    if action_index == 2:
        return PolicyAction.DUCK
    return PolicyAction.NOOP


def default_neat_config_text(population: int, seed: int) -> str:
    return f"""[NEAT]
fitness_criterion     = max
fitness_threshold     = 500.0
pop_size              = {population}
reset_on_extinction   = False
no_fitness_termination = False

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.5
conn_delete_prob        = 0.3
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True
initial_connection      = full_direct
node_add_prob           = 0.2
node_delete_prob        = 0.2
num_hidden              = 0
num_inputs              = {OBSERVATION_SIZE}
num_outputs             = 3
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 8
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2

[DinoIA]
seed = {seed}
"""


class NeatTrainingArtifacts:
    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir) if output_dir is not None else DEFAULT_ARTIFACTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "training_history.json"
        self.summary_path = self.output_dir / "latest_summary.txt"
        self.best_genome_path = self.output_dir / "best_genome.pkl"
        self.winner_path = self.output_dir / "winner.pkl"
        self.config_path = self.output_dir / "neat_config.txt"

    def ensure_config_file(self, neat_config: NeatConfig) -> Path:
        text = default_neat_config_text(neat_config.population, neat_config.seed)
        self.config_path.write_text(text, encoding="utf-8")
        return self.config_path

    def read_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        raw = json.loads(self.history_path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else []

    def append_generation(self, entry: dict[str, Any]) -> None:
        history = self.read_history()
        history.append(entry)
        self.history_path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
        self.summary_path.write_text(format_latest_summary(entry), encoding="utf-8")

    def save_genome(self, path: Path, genome: Any) -> None:
        with path.open("wb") as handle:
            pickle.dump(genome, handle)


def format_latest_summary(entry: dict[str, Any]) -> str:
    return "\n".join(
        [
            "DinoIA NEAT latest summary",
            f"generation={entry.get('generation')}",
            f"best_fitness={entry.get('best_fitness')}",
            f"mean_fitness={entry.get('mean_fitness')}",
            f"best_distance_px={entry.get('best_distance_px')}",
            f"best_genome_id={entry.get('best_genome_id')}",
            f"winner_path={entry.get('winner_path')}",
            f"checkpoint_path={entry.get('checkpoint_path')}",
        ]
    )


class RealNeatEvaluator:
    def __init__(
        self,
        *,
        neat_config: NeatConfig | None = None,
        real_config: RealGameConfig | None = None,
        vision_config: DinoVisionConfig | None = None,
        sim_config: SimConfig | None = None,
        capture: DinoScreenCapture | None = None,
        analyzer: DinoVisionAnalyzer | None = None,
        controller: KeyboardController | None = None,
        gym_teacher_policy: LinearPolicy | None = None,
    ):
        self.neat_config = neat_config or NeatConfig()
        self.real_config = real_config or RealGameConfig(debug=False)
        self.vision_config = vision_config or DinoVisionConfig()
        self.sim_config = sim_config or SimConfig()
        self.capture = capture or DinoScreenCapture(self.real_config)
        self.analyzer = analyzer or DinoVisionAnalyzer(self.vision_config)
        self.controller = controller or KeyboardController()
        self.observation_builder = RealRLObservationBuilder(self.sim_config)
        self.action_gate = RealActionGate(min_action_interval_s=self.vision_config.min_action_interval_s)
        self.gym_teacher_policy = gym_teacher_policy or self._load_gym_teacher_policy(self.neat_config.teacher_source)

    @staticmethod
    def _load_gym_teacher_policy(source: str | Path | None) -> LinearPolicy | None:
        if source is not None:
            path = Path(source)
            if path.exists():
                try:
                    return LinearPolicy.load(path)
                except Exception:
                    pass
        return load_best_gym_policy()

    def close(self) -> None:
        close = getattr(self.capture, "close", None)
        if callable(close):
            close()

    def evaluate_network(self, network: Any, *, genome_id: int, generation: int) -> NeatEpisodeResult:
        self.action_gate.last_jump_at = 0.0
        reset_result = self.prepare_episode()
        if not reset_result.ready:
            print(
                "evolve-real: reset warning reason={reason} attempts={attempts} waited_s={waited:.1f}".format(
                    reason=reset_result.reason,
                    attempts=reset_result.attempts,
                    waited=reset_result.waited_seconds,
                ),
                flush=True,
            )

        start = time.time()
        frame_interval = 1.0 / max(1.0, self.real_config.frame_rate)
        previous_state: VisionState | None = None
        fitness = 0.0
        distance_px = 0.0
        passed_obstacles = 0
        jumps = ducks = noops = frames = 0
        collision = False
        terminal_reason = "timeout"
        risk_seen = False
        last_motion_at = start
        last_distance: float | None = None
        pending_pass: PendingPassState | None = None
        previous_frame_at = start

        try:
            while distance_px < self.neat_config.episode_distance_px and frames < self.neat_config.max_episode_frames:
                now = time.time()
                frame_dt = max(0.0, now - previous_frame_at)
                state = self._capture_state(now)
                if state.game_over:
                    collision = True
                    terminal_reason = "game_over"
                    fitness += self.neat_config.collision_penalty
                    break
                speed_px_s = max(0.0, self.observation_builder.estimate_current_speed_px_s(state))
                if speed_px_s <= 1.0 and not state.game_over:
                    speed_px_s = max(speed_px_s, 200.0)
                distance_px += speed_px_s * frame_dt
                if pending_pass is not None:
                    if pending_pass.label == "cactus" and not pending_pass.airborne_seen and not self._is_grounded(state):
                        pending_pass.airborne_seen = True
                    if pending_pass.label == "bird" and not pending_pass.ducking_seen and self._is_ducking(state):
                        pending_pass.ducking_seen = True
                if self._confirmed_pass(pending_pass, state, now):
                    fitness += self.neat_config.pass_reward
                    passed_obstacles += 1
                    pending_pass = None
                    risk_seen = False
                    last_motion_at = now
                if self._passed_obstacle(previous_state, state):
                    risk_seen = False
                    last_motion_at = now
                    if previous_state is not None and previous_state.nearest_obstacle is not None:
                        pending_pass = PendingPassState(
                            started_at=now,
                            label=previous_state.nearest_obstacle.label,
                            rect_x=previous_state.nearest_obstacle.rect.x,
                            rect_right=previous_state.nearest_obstacle.rect.right,
                            speed_px_s=max(0.0, previous_state.estimated_speed_px_s),
                            player_x=previous_state.player_box.x,
                            airborne_seen=not self._is_grounded(previous_state),
                            ducking_seen=self._is_ducking(previous_state),
                        )
                if self._danger_seen(state):
                    risk_seen = True
                if self._has_motion(state, last_distance):
                    last_motion_at = now
                if risk_seen and now - last_motion_at >= self.neat_config.stalled_death_seconds:
                    collision = True
                    terminal_reason = "stalled_after_danger"
                    fitness += self.neat_config.collision_penalty
                    break

                obs = self.observation_builder.build(state)
                requested_action = action_from_outputs(network.activate(obs.tolist()))
                teacher_action = self._teacher_action(state)
                gym_teacher_action = self._gym_teacher_action(state)
                chosen_action = self._apply_safety_override(requested_action, state, teacher_action, gym_teacher_action)
                action = self._filter_action(chosen_action, state, now)
                self.controller.perform(action)
                if action is PolicyAction.JUMP:
                    jumps += 1
                elif action is PolicyAction.DUCK:
                    ducks += 1
                else:
                    noops += 1

                fitness += self._reward(requested_action, action, previous_state, state, gym_teacher_action)

                frames += 1
                previous_state = state
                last_distance = state.nearest_distance_px
                if self.real_config.debug:
                    self._show_overlay(
                        state,
                        action,
                        fitness,
                        distance_px,
                        genome_id,
                        generation,
                        terminal_reason,
                        teacher_action,
                        gym_teacher_action,
                    )

                print(
                    "evolve-real: generation={generation} genome={genome} fitness={fitness:.3f} "
                    "distance_px={distance:.0f} passed={passed} action={action} teacher={teacher} gym_teacher={gym_teacher} status={status}".format(
                        generation=generation,
                        genome=genome_id,
                        fitness=fitness,
                        distance=distance_px,
                        passed=passed_obstacles,
                        action=action.value,
                        teacher=teacher_action.value,
                        gym_teacher=gym_teacher_action.value if gym_teacher_action is not None else "none",
                        status="risk" if risk_seen else "running",
                    ),
                    flush=True,
                )
                previous_frame_at = now
                time.sleep(max(0.0, frame_interval - (time.time() - now)))
        finally:
            try:
                self.controller.release_duck()
                if collision and self.real_config.auto_restart:
                    self.prepare_episode()
                if self.real_config.debug:
                    cv2.destroyAllWindows()
            finally:
                self.close()

        return NeatEpisodeResult(
            genome_id=genome_id,
            generation=generation,
            fitness=float(fitness),
            alive_seconds=time.time() - start,
            distance_px=float(distance_px),
            frames=frames,
            passed_obstacles=passed_obstacles,
            jumps=jumps,
            ducks=ducks,
            noops=noops,
            collision=collision,
            terminal_reason=terminal_reason,
        )

    def _capture_state(self, timestamp: float) -> VisionState:
        frame = self.capture.capture()
        return self.analyzer.analyze(frame, timestamp=timestamp)

    def prepare_episode(self) -> EpisodeResetResult:
        self.controller.release_duck()
        if self.real_config.focus_window:
            try:
                self.capture.focus_window()
            except Exception:
                pass

        attempts = 0
        start = time.time()
        deadline = start + self.neat_config.reset_timeout_seconds
        last_state: VisionState | None = None

        while time.time() < deadline:
            state = self._capture_state(time.time())
            last_state = state
            if not state.game_over:
                if self.real_config.auto_start and attempts == 0:
                    self.controller.restart()
                    attempts += 1
                    time.sleep(0.25)
                    continue
                return EpisodeResetResult(
                    ready=True,
                    attempts=attempts,
                    waited_seconds=time.time() - start,
                    reason="ready",
                )

            self.controller.restart()
            attempts += 1
            time.sleep(self.neat_config.reset_poll_seconds)

        return EpisodeResetResult(
            ready=False,
            attempts=attempts,
            waited_seconds=time.time() - start,
            reason="still_game_over" if last_state is not None and last_state.game_over else "timeout",
        )

    def _filter_action(self, action: PolicyAction, state: VisionState, now: float) -> PolicyAction:
        grounded = state.player_box.bottom >= state.ground_y - 4
        return self.action_gate.filter(action, grounded=grounded, now=now)

    def _reward(
        self,
        requested_action: PolicyAction,
        executed_action: PolicyAction,
        previous_state: VisionState | None,
        state: VisionState,
        gym_teacher_action: PolicyAction | None = None,
    ) -> float:
        reward = self.neat_config.survival_reward
        if executed_action is not PolicyAction.NOOP:
            reward += self.neat_config.action_penalty
        if requested_action is not executed_action:
            reward += self.neat_config.unnecessary_jump_penalty
        if self._bad_action(requested_action, state):
            reward += self.neat_config.unnecessary_jump_penalty
        if self._missed_action(requested_action, state):
            reward += self.neat_config.missed_action_penalty
        reward += self._teacher_reward(requested_action, state)
        reward += self._gym_teacher_reward(requested_action, gym_teacher_action)
        return reward

    def _teacher_reward(self, action: PolicyAction, state: VisionState) -> float:
        teacher_action = self._teacher_action(state)
        if teacher_action is PolicyAction.NOOP:
            return 0.0
        if action is teacher_action:
            return self.neat_config.teacher_match_reward
        return self.neat_config.teacher_mismatch_penalty

    def _teacher_action(self, state: VisionState) -> PolicyAction:
        return DinoRulePolicy(self.vision_config).decide(state, now=state.timestamp).action

    def _gym_teacher_action(self, state: VisionState) -> PolicyAction | None:
        if self.gym_teacher_policy is None:
            return None
        obs = self.observation_builder.build(state)
        action_index = self.gym_teacher_policy.act_one(obs)
        if action_index == 1:
            return PolicyAction.JUMP
        if action_index == 2:
            return PolicyAction.DUCK
        return PolicyAction.NOOP

    def _gym_teacher_reward(self, action: PolicyAction, gym_teacher_action: PolicyAction | None) -> float:
        if gym_teacher_action is None or gym_teacher_action is PolicyAction.NOOP:
            return 0.0
        if action is gym_teacher_action:
            return self.neat_config.teacher_match_reward
        return self.neat_config.teacher_mismatch_penalty

    def _apply_safety_override(
        self,
        requested_action: PolicyAction,
        state: VisionState,
        teacher_action: PolicyAction,
        gym_teacher_action: PolicyAction | None,
    ) -> PolicyAction:
        if not self.neat_config.enable_safety_overrides:
            return requested_action
        if not self._danger_seen(state):
            return requested_action
        if requested_action is not PolicyAction.NOOP and not self._bad_action(requested_action, state):
            return requested_action
        if teacher_action is not PolicyAction.NOOP:
            return teacher_action
        if gym_teacher_action is not None and gym_teacher_action is not PolicyAction.NOOP:
            return gym_teacher_action
        return requested_action

    def _passed_obstacle(self, previous_state: VisionState | None, state: VisionState) -> bool:
        if previous_state is None:
            return False
        previous_distance = previous_state.nearest_distance_px
        if previous_state.nearest_obstacle is None or previous_distance is None:
            return False
        if previous_distance > self.neat_config.pass_detection_distance_px:
            return False
        current_distance = state.nearest_distance_px
        if state.nearest_obstacle is None or current_distance is None:
            return True
        return current_distance > previous_distance + self.neat_config.pass_detection_distance_px

    def _confirmed_pass(
        self,
        pending_pass: PendingPassState | None,
        state: VisionState,
        now: float,
    ) -> bool:
        if pending_pass is None:
            return False
        if state.game_over:
            return False
        if now - pending_pass.started_at < self.neat_config.pass_confirmation_seconds:
            return False
        if not self._has_motion(state, None):
            return False
        if pending_pass.label == "cactus" and not pending_pass.airborne_seen:
            return False
        if pending_pass.label == "bird" and not pending_pass.ducking_seen:
            return False

        elapsed = now - pending_pass.started_at
        estimated_right = pending_pass.rect_right - pending_pass.speed_px_s * elapsed
        return estimated_right <= state.player_box.x - self.neat_config.pass_cross_margin_px

    def _danger_seen(self, state: VisionState) -> bool:
        nearest = state.nearest_obstacle
        distance = state.nearest_distance_px
        if nearest is None or distance is None:
            return False
        if nearest.label == "bird":
            return distance <= self.neat_config.duck_bird_distance_px
        return distance <= self._jump_trigger_distance(state)

    def _has_motion(self, state: VisionState, last_distance: float | None) -> bool:
        if state.estimated_speed_px_s > self.neat_config.stalled_speed_px_s:
            return True
        if last_distance is None or state.nearest_distance_px is None:
            return False
        return abs(state.nearest_distance_px - last_distance) > 3.0

    def _is_grounded(self, state: VisionState) -> bool:
        return state.player_box.bottom >= state.ground_y - 4

    def _is_ducking(self, state: VisionState) -> bool:
        return self._is_grounded(state) and state.player_box.h <= self.vision_config.max_ducking_player_height_px + 4

    def _bad_action(self, action: PolicyAction, state: VisionState) -> bool:
        nearest = state.nearest_obstacle
        distance = state.nearest_distance_px
        grounded = state.player_box.bottom >= state.ground_y - 4
        if action is PolicyAction.JUMP:
            return (not grounded) or nearest is None or distance is None or distance > self.neat_config.safe_jump_distance_px
        if action is PolicyAction.DUCK:
            return not (
                grounded
                and nearest is not None
                and nearest.label == "bird"
                and distance is not None
                and distance <= self.neat_config.duck_bird_distance_px
            )
        return False

    def _missed_action(self, action: PolicyAction, state: VisionState) -> bool:
        if action is not PolicyAction.NOOP:
            return False
        nearest = state.nearest_obstacle
        distance = state.nearest_distance_px
        if nearest is None or distance is None:
            return False
        if nearest.label == "bird":
            return distance <= self.neat_config.duck_bird_distance_px
        return distance <= self._jump_trigger_distance(state)

    def _jump_trigger_distance(self, state: VisionState) -> float:
        return self.vision_config.jump_base_distance_px + min(
            state.estimated_speed_px_s * self.vision_config.jump_speed_factor,
            self.vision_config.jump_base_distance_px * 2.2,
        )

    def _show_overlay(
        self,
        state: VisionState,
        action: PolicyAction,
        fitness: float,
        distance_px: float,
        genome_id: int,
        generation: int,
        terminal_reason: str = "running",
        teacher_action: PolicyAction = PolicyAction.NOOP,
        gym_teacher_action: PolicyAction | None = None,
    ) -> None:
        overlay = self.analyzer.draw_overlay(state)
        cv2.putText(
            overlay,
            f"gen={generation} genome={genome_id} action={action.value} teacher={teacher_action.value} sim={gym_teacher_action.value if gym_teacher_action is not None else 'none'} fitness={fitness:.2f} dist={distance_px:.0f}",
            (10, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"status={terminal_reason}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("DinoIA - NEAT Real", overlay)
        cv2.waitKey(1)


def _load_neat():
    try:
        import neat
    except ImportError as exc:
        raise ImportError("Install neat-python first: pip install neat-python==2.0.0") from exc
    return neat


def checkpoint_generation_start(path: str | Path | None) -> int:
    if path is None:
        return 0
    match = re.search(r"neat-checkpoint-(\d+)$", str(path))
    if match is None:
        return 0
    return int(match.group(1)) + 1


def evolve_real_neat(
    *,
    neat_config: NeatConfig | None = None,
    real_config: RealGameConfig | None = None,
    vision_config: DinoVisionConfig | None = None,
    sim_config: SimConfig | None = None,
    resume_checkpoint: str | Path | None = None,
) -> Path:
    neat_config = neat_config or NeatConfig()
    artifacts = NeatTrainingArtifacts(neat_config.output_dir)
    config_path = artifacts.ensure_config_file(neat_config)
    neat = _load_neat()
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        population = neat.Checkpointer.restore_checkpoint(str(resume_path))
        print(
            f"evolve-real: resumed checkpoint={resume_path} additional_generations={neat_config.generations}",
            flush=True,
        )
        print("evolve-real: checkpoint population is preserved; --population is ignored while resuming", flush=True)
    else:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(config_path),
        )
        population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.Checkpointer(1, filename_prefix=str(artifacts.output_dir / "neat-checkpoint-")))

    generation_index = {"value": checkpoint_generation_start(resume_checkpoint)}
    best_genome: Any | None = None
    best_fitness = float("-inf")

    def eval_genomes(genomes, neat_runtime_config):
        nonlocal best_genome, best_fitness
        generation = generation_index["value"]
        evaluator = RealNeatEvaluator(
            neat_config=neat_config,
            real_config=real_config,
            vision_config=vision_config,
            sim_config=sim_config,
        )
        results: list[NeatEpisodeResult] = []
        for genome_id, genome in genomes:
            network = neat.nn.FeedForwardNetwork.create(genome, neat_runtime_config)
            result = evaluator.evaluate_network(network, genome_id=genome_id, generation=generation)
            genome.fitness = result.fitness
            results.append(result)
            if result.fitness > best_fitness:
                best_fitness = result.fitness
                best_genome = genome
                artifacts.save_genome(artifacts.best_genome_path, genome)

        fitnesses = [result.fitness for result in results]
        best_result = max(results, key=lambda item: item.fitness) if results else None
        entry = {
            "generation": generation,
            "best_fitness": max(fitnesses) if fitnesses else 0.0,
            "mean_fitness": statistics.fmean(fitnesses) if fitnesses else 0.0,
            "best_distance_px": best_result.distance_px if best_result is not None else 0.0,
            "best_genome_id": best_result.genome_id if best_result is not None else None,
            "winner_path": str(artifacts.winner_path),
            "checkpoint_path": str(artifacts.output_dir / f"neat-checkpoint-{generation}"),
            "episodes": [asdict(result) for result in results],
        }
        artifacts.append_generation(entry)
        print(
            "evolve-real: generation={generation} complete best={best:.3f} mean={mean:.3f} distance_px={distance:.0f}".format(
                generation=generation,
                best=entry["best_fitness"],
                mean=entry["mean_fitness"],
                distance=entry["best_distance_px"],
            ),
            flush=True,
        )
        generation_index["value"] += 1

    try:
        winner = population.run(eval_genomes, neat_config.generations)
    except KeyboardInterrupt:
        if best_genome is None:
            raise
        winner = best_genome
        print("evolve-real: interrupted, saving best genome so far", flush=True)

    artifacts.save_genome(artifacts.winner_path, winner)
    return artifacts.winner_path


class DinoRealNeatRunner:
    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        real_config: RealGameConfig | None = None,
        vision_config: DinoVisionConfig | None = None,
        sim_config: SimConfig | None = None,
        neat_config: NeatConfig | None = None,
    ):
        self.neat_config = neat_config or NeatConfig()
        self.real_config = real_config or RealGameConfig(debug=False)
        self.vision_config = vision_config or DinoVisionConfig()
        self.sim_config = sim_config or SimConfig()
        self.model_path = Path(model_path) if model_path is not None else select_best_model(self.neat_config.output_dir)
        if self.model_path is None:
            raise FileNotFoundError("No NEAT genome was found in artifacts/neat. Run evolve-real first.")
        with Path(self.model_path).open("rb") as handle:
            self.genome = pickle.load(handle)

        neat = _load_neat()
        artifacts = NeatTrainingArtifacts(self.neat_config.output_dir)
        config_path = artifacts.config_path
        if not config_path.exists():
            config_path = artifacts.ensure_config_file(self.neat_config)
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(config_path),
        )
        self.network = neat.nn.FeedForwardNetwork.create(self.genome, config)
        self.evaluator = RealNeatEvaluator(
            neat_config=self.neat_config,
            real_config=self.real_config,
            vision_config=self.vision_config,
            sim_config=self.sim_config,
        )

    def run(self, duration_s: float | None = None) -> NeatRunResult:
        start = time.time()
        steps = jumps = ducks = noops = safety_overrides = 0
        self.evaluator.prepare_episode()
        try:
            while duration_s is None or time.time() - start < duration_s:
                now = time.time()
                state = self.evaluator._capture_state(now)
                if self.real_config.auto_restart and state.game_over:
                    self.evaluator.prepare_episode()
                    continue
                obs = self.evaluator.observation_builder.build(state)
                requested_action = action_from_outputs(self.network.activate(obs.tolist()))
                teacher_action = self.evaluator._teacher_action(state)
                gym_teacher_action = self.evaluator._gym_teacher_action(state)
                chosen_action = self.evaluator._apply_safety_override(
                    requested_action,
                    state,
                    teacher_action,
                    gym_teacher_action,
                )
                if chosen_action is not requested_action:
                    safety_overrides += 1
                action = self.evaluator._filter_action(chosen_action, state, now)
                self.evaluator.controller.perform(action)
                if action is PolicyAction.JUMP:
                    jumps += 1
                elif action is PolicyAction.DUCK:
                    ducks += 1
                else:
                    noops += 1
                steps += 1
                if self.real_config.debug:
                    self.evaluator._show_overlay(
                        state,
                        action,
                        0.0,
                        0.0,
                        0,
                        0,
                        teacher_action=teacher_action,
                        gym_teacher_action=gym_teacher_action,
                    )
                time.sleep(1.0 / max(1.0, self.real_config.frame_rate))
        finally:
            try:
                self.evaluator.controller.release_duck()
                if self.real_config.debug:
                    cv2.destroyAllWindows()
            finally:
                self.evaluator.close()

        return NeatRunResult(
            model_path=str(self.model_path),
            steps=steps,
            jumps=jumps,
            ducks=ducks,
            noops=noops,
            safety_overrides=safety_overrides,
        )
