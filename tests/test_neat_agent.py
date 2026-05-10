from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np

from dinoia.config import NeatConfig, RealGameConfig
from dinoia.gym_parallel import LinearPolicy
from dinoia.neat_agent import (
    NeatTrainingArtifacts,
    PendingPassState,
    RealNeatEvaluator,
    action_from_outputs,
    checkpoint_generation_start,
    format_latest_summary,
)
from dinoia.types import ObstacleDetection, PolicyAction, Rect, VisionState


def _state(distance=None, *, label="cactus", game_over=False, player_y=120):
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    player = Rect(40, player_y, 30, 44)
    obstacle = None
    obstacles = []
    if distance is not None:
        obstacle = ObstacleDetection(
            rect=Rect(player.right + int(distance), 122, 24, 40),
            label=label,
            distance_px=float(distance),
        )
        obstacles = [obstacle]
    return VisionState(
        timestamp=1.0,
        frame=frame,
        playfield=frame,
        player_box=player,
        ground_y=164,
        obstacles=obstacles,
        nearest_obstacle=obstacle,
        nearest_distance_px=float(distance) if distance is not None else None,
        estimated_speed_px_s=320.0,
        estimated_player_vy_px_s=0.0,
        game_over=game_over,
    )


class FakeCapture:
    def capture(self):
        return np.zeros((200, 400, 3), dtype=np.uint8)

    def focus_window(self):
        pass


class FakeAnalyzer:
    def __init__(self, states):
        self.states = list(states)

    def analyze(self, frame, timestamp=None):
        if self.states:
            state = self.states.pop(0)
            state.timestamp = timestamp or state.timestamp
            return state
        return _state(None)


class FakeController:
    def __init__(self):
        self.restarts = 0
        self.releases = 0

    def release_duck(self):
        self.releases += 1

    def restart(self):
        self.restarts += 1

    def perform(self, action):
        return None


def test_neat_output_maps_to_actions():
    assert action_from_outputs([0.9, 0.1, 0.2]) is PolicyAction.NOOP
    assert action_from_outputs([0.1, 0.9, 0.2]) is PolicyAction.JUMP
    assert action_from_outputs([0.1, 0.2, 0.9]) is PolicyAction.DUCK


def test_neat_checkpoint_generation_start_uses_next_generation():
    assert checkpoint_generation_start(Path("artifacts/neat/neat-checkpoint-7")) == 8
    assert checkpoint_generation_start(Path("artifacts/neat/other-file")) == 0
    assert checkpoint_generation_start(None) == 0


def test_neat_detects_pass_candidate_without_immediate_reward():
    evaluator = RealNeatEvaluator(
        neat_config=NeatConfig(),
        real_config=RealGameConfig(focus_window=False, debug=False),
    )
    reward = evaluator._reward(PolicyAction.NOOP, PolicyAction.NOOP, _state(40), _state(None))

    assert evaluator._passed_obstacle(_state(40), _state(None)) is True
    assert reward == evaluator.neat_config.survival_reward


def test_neat_confirms_pass_only_when_scene_keeps_moving():
    evaluator = RealNeatEvaluator(
        neat_config=NeatConfig(stalled_speed_px_s=25.0),
        real_config=RealGameConfig(focus_window=False, debug=False),
    )
    moving = _state(160)
    moving.estimated_speed_px_s = 100.0
    moving_without_next_obstacle = _state(None)
    moving_without_next_obstacle.estimated_speed_px_s = 100.0
    stalled = _state(None)
    stalled.estimated_speed_px_s = 0.0
    game_over = _state(None, game_over=True)
    pending_cactus = PendingPassState(
        started_at=1.0,
        label="cactus",
        rect_x=120,
        rect_right=160,
        speed_px_s=320.0,
        player_x=40,
        airborne_seen=True,
    )
    grounded_cactus = PendingPassState(
        started_at=1.0,
        label="cactus",
        rect_x=120,
        rect_right=160,
        speed_px_s=320.0,
        player_x=40,
        airborne_seen=False,
    )

    assert evaluator._confirmed_pass(pending_cactus, moving, now=1.5) is True
    assert evaluator._confirmed_pass(pending_cactus, moving, now=1.1) is False
    assert evaluator._confirmed_pass(pending_cactus, moving_without_next_obstacle, now=1.5) is True
    assert evaluator._confirmed_pass(grounded_cactus, moving_without_next_obstacle, now=1.5) is False
    assert evaluator._confirmed_pass(pending_cactus, stalled, now=1.5) is False
    assert evaluator._confirmed_pass(pending_cactus, game_over, now=1.5) is False
    assert evaluator._confirmed_pass(None, moving, now=1.5) is False


def test_neat_fitness_penalizes_collision_and_missed_action():
    evaluator = RealNeatEvaluator(
        neat_config=NeatConfig(),
        real_config=RealGameConfig(focus_window=False, debug=False),
    )
    danger = _state(30)
    reward = evaluator._reward(PolicyAction.NOOP, PolicyAction.NOOP, None, danger)
    collision_total = reward + evaluator.neat_config.collision_penalty

    assert reward < evaluator.neat_config.survival_reward
    assert collision_total < reward


def test_neat_teacher_rewards_matching_danger_action():
    evaluator = RealNeatEvaluator(
        neat_config=NeatConfig(),
        real_config=RealGameConfig(focus_window=False, debug=False),
    )
    danger = _state(30)

    matching = evaluator._teacher_reward(PolicyAction.JUMP, danger)
    mismatching = evaluator._teacher_reward(PolicyAction.NOOP, danger)

    assert matching == evaluator.neat_config.teacher_match_reward
    assert mismatching == evaluator.neat_config.teacher_mismatch_penalty


def test_neat_uses_gym_teacher_policy_as_guidance():
    teacher_policy = LinearPolicy(
        weights=np.zeros((3, 13), dtype=np.float32),
        bias=np.array([0.0, 5.0, 0.0], dtype=np.float32),
    )
    evaluator = RealNeatEvaluator(
        neat_config=NeatConfig(),
        real_config=RealGameConfig(focus_window=False, debug=False),
        gym_teacher_policy=teacher_policy,
    )
    danger = _state(30)

    teacher_action = evaluator._gym_teacher_action(danger)
    reward = evaluator._gym_teacher_reward(PolicyAction.JUMP, teacher_action)

    assert teacher_action is PolicyAction.JUMP
    assert reward == evaluator.neat_config.teacher_match_reward


def test_neat_can_load_teacher_policy_from_config_path():
    output_dir = Path("artifacts") / f"test-teacher-{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "best_policy.npz"
    teacher_policy = LinearPolicy(
        weights=np.zeros((3, 13), dtype=np.float32),
        bias=np.array([0.0, 5.0, 0.0], dtype=np.float32),
    )
    np.savez_compressed(policy_path, weights=teacher_policy.weights, bias=teacher_policy.bias)

    evaluator = RealNeatEvaluator(
        neat_config=NeatConfig(teacher_source=str(policy_path)),
        real_config=RealGameConfig(focus_window=False, debug=False),
    )
    danger = _state(30)

    assert evaluator.gym_teacher_policy is not None
    assert evaluator._gym_teacher_action(danger) is PolicyAction.JUMP


def test_neat_safety_override_uses_teacher_when_model_noops_in_danger():
    evaluator = RealNeatEvaluator(
        neat_config=NeatConfig(),
        real_config=RealGameConfig(focus_window=False, debug=False),
    )
    danger = _state(30)

    action = evaluator._apply_safety_override(
        PolicyAction.NOOP,
        danger,
        PolicyAction.JUMP,
        None,
    )

    assert action is PolicyAction.JUMP


def test_neat_safety_override_keeps_valid_model_action():
    evaluator = RealNeatEvaluator(
        neat_config=NeatConfig(),
        real_config=RealGameConfig(focus_window=False, debug=False),
    )
    danger = _state(30)

    action = evaluator._apply_safety_override(
        PolicyAction.JUMP,
        danger,
        PolicyAction.DUCK,
        None,
    )

    assert action is PolicyAction.JUMP


def test_neat_detects_danger_and_stalled_scene():
    evaluator = RealNeatEvaluator(
        neat_config=NeatConfig(stalled_speed_px_s=25.0),
        real_config=RealGameConfig(focus_window=False, debug=False),
    )
    danger = _state(30)
    stalled = _state(30)
    stalled.estimated_speed_px_s = 0.0
    moving = _state(20)
    moving.estimated_speed_px_s = 80.0

    assert evaluator._danger_seen(danger) is True
    assert evaluator._has_motion(stalled, last_distance=30.0) is False
    assert evaluator._has_motion(moving, last_distance=30.0) is True


def test_neat_prepare_episode_restarts_until_game_over_clears():
    controller = FakeController()
    evaluator = RealNeatEvaluator(
        neat_config=NeatConfig(reset_timeout_seconds=1.0, reset_poll_seconds=0.0),
        real_config=RealGameConfig(focus_window=False, debug=False, auto_start=True),
        capture=FakeCapture(),
        analyzer=FakeAnalyzer([_state(None, game_over=True), _state(None, game_over=False)]),
        controller=controller,
    )

    result = evaluator.prepare_episode()

    assert result.ready is True
    assert controller.restarts >= 1
    assert controller.releases == 1


def test_neat_artifacts_persist_history_summary_and_genome():
    output_dir = Path("artifacts") / f"test-neat-{uuid.uuid4().hex}"
    artifacts = NeatTrainingArtifacts(output_dir)
    entry = {
        "generation": 1,
        "best_fitness": 4.2,
        "mean_fitness": 1.7,
        "best_genome_id": 9,
        "winner_path": str(output_dir / "winner.pkl"),
        "checkpoint_path": str(output_dir / "neat-checkpoint-1"),
    }

    artifacts.append_generation(entry)
    artifacts.save_genome(artifacts.best_genome_path, {"genome": 1})

    history = json.loads(artifacts.history_path.read_text(encoding="utf-8"))
    assert history == [entry]
    assert "best_fitness=4.2" in artifacts.summary_path.read_text(encoding="utf-8")
    assert artifacts.best_genome_path.exists()


def test_neat_latest_summary_format_contains_visible_progress():
    summary = format_latest_summary(
        {
            "generation": 2,
            "best_fitness": 7.0,
            "mean_fitness": 3.5,
            "best_genome_id": 4,
            "winner_path": "winner.pkl",
            "checkpoint_path": "neat-checkpoint-2",
        }
    )

    assert "generation=2" in summary
    assert "best_fitness=7.0" in summary
