from __future__ import annotations

import platform
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import SimConfig, TrainingConfig
from .project_state import DEFAULT_ARTIFACTS_DIR, select_best_model, select_latest_model, summarize_project


@dataclass(slots=True)
class DoctorCheck:
    name: str
    status: str
    detail: str


@dataclass(slots=True)
class DoctorReport:
    checks: list[DoctorCheck]

    @property
    def ok(self) -> bool:
        return all(check.status != "ERROR" for check in self.checks)


def _check_import(module_name: str) -> tuple[bool, str]:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        return True, f"{module_name}={version}"
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def run_doctor(output_dir: str | Path | None = None) -> DoctorReport:
    checks: list[DoctorCheck] = []

    checks.append(DoctorCheck("Python", "OK", platform.python_version()))
    checks.append(DoctorCheck("Platform", "OK", platform.platform()))

    for module_name in ("numpy", "cv2", "gymnasium", "stable_baselines3", "mss", "pyautogui", "pygetwindow"):
        ok, detail = _check_import(module_name)
        checks.append(DoctorCheck(module_name, "OK" if ok else "ERROR", detail))

    torch_ok, torch_detail = _check_import("torch")
    if torch_ok:
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            cuda_version = getattr(torch.version, "cuda", None)
            device_name = torch.cuda.get_device_name(0) if cuda_available else "n/a"
            detail = f"version={torch.__version__} cuda={cuda_available} cuda_version={cuda_version} device={device_name}"
            checks.append(DoctorCheck("torch", "OK", detail))
        except Exception as exc:
            checks.append(DoctorCheck("torch", "ERROR", f"{exc.__class__.__name__}: {exc}"))
    else:
        checks.append(DoctorCheck("torch", "ERROR", torch_detail))

    artifacts_root = Path(output_dir) if output_dir is not None else DEFAULT_ARTIFACTS_DIR
    try:
        artifacts_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=artifacts_root, delete=True):
            pass
        checks.append(DoctorCheck("Artifacts write", "OK", str(artifacts_root)))
    except Exception as exc:
        checks.append(DoctorCheck("Artifacts write", "ERROR", f"{exc.__class__.__name__}: {exc}"))

    latest_model = select_latest_model(artifacts_root)
    best_model = select_best_model(artifacts_root)
    if latest_model is not None:
        checks.append(DoctorCheck("Latest model", "OK", str(latest_model)))
    else:
        checks.append(DoctorCheck("Latest model", "WARN", "none found"))

    if best_model is not None:
        checks.append(DoctorCheck("Best model", "OK", str(best_model)))
    else:
        checks.append(DoctorCheck("Best model", "WARN", "none found"))

    report = summarize_project(artifacts_root)
    training = report.training_config or {}
    observation_size = training.get("training", {}).get("observation_size") if isinstance(training, dict) else None
    expected_observation_size = TrainingConfig().observation_size
    if observation_size is None:
        checks.append(DoctorCheck("Checkpoint compatibility", "WARN", "no observation_size metadata yet"))
    elif observation_size != expected_observation_size:
        checks.append(
            DoctorCheck(
                "Checkpoint compatibility",
                "WARN",
                f"saved={observation_size} expected={expected_observation_size}",
            )
        )
    else:
        checks.append(DoctorCheck("Checkpoint compatibility", "OK", f"observation_size={observation_size}"))

    return DoctorReport(checks=checks)


def format_doctor_report(report: DoctorReport) -> str:
    lines = ["doctor: environment check"]
    for check in report.checks:
        lines.append(f"- {check.name}: {check.status} ({check.detail})")
    return "\n".join(lines)


def benchmark_training(
    *,
    timesteps: int = 3_000,
    devices: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from .config import SimConfig
    from .sim.env import DinoSimEnv

    results: list[dict[str, Any]] = []
    device_list = devices or ["cpu"]
    try:
        import torch

        if torch.cuda.is_available() and "cuda" not in device_list:
            device_list = ["cpu", "cuda"]
    except Exception:
        pass

    quick_params = dict(
        learning_rate=2e-2,
        gamma=0.99,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.45,
        exploration_final_eps=0.08,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=42,
        verbose=0,
    )

    for device in device_list:
        train_env = DummyVecEnv([lambda: Monitor(DinoSimEnv(config=SimConfig(seed=42), render_mode=None))])
        start = time.perf_counter()
        try:
            model = DQN("MlpPolicy", train_env, device=device, **quick_params)
            model.learn(total_timesteps=timesteps, progress_bar=False)
            elapsed = time.perf_counter() - start
            results.append(
                {
                    "device": device,
                    "elapsed_s": elapsed,
                    "timesteps": timesteps,
                    "timesteps_per_second": timesteps / max(1e-6, elapsed),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "device": device,
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "timesteps": timesteps,
                }
            )
        finally:
            train_env.close()
    return results
