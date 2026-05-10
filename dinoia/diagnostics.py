from __future__ import annotations

import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .observations import OBSERVATION_SIZE
from .ppo_agent import DEFAULT_PPO_ARTIFACTS_DIR, summarize_ppo


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

    for module_name in (
        "numpy",
        "cv2",
        "gymnasium",
        "stable_baselines3",
        "torch",
        "neat",
        "mss",
        "pyautogui",
        "pygetwindow",
    ):
        ok, detail = _check_import(module_name)
        checks.append(DoctorCheck(module_name, "OK" if ok else "ERROR", detail))

    artifacts_root = Path(output_dir) if output_dir is not None else DEFAULT_PPO_ARTIFACTS_DIR
    try:
        artifacts_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=artifacts_root, delete=True):
            pass
        checks.append(DoctorCheck("Artifacts write", "OK", str(artifacts_root)))
    except Exception as exc:
        checks.append(DoctorCheck("Artifacts write", "ERROR", f"{exc.__class__.__name__}: {exc}"))

    summary = summarize_ppo(artifacts_root)
    if summary.best_model_path is not None:
        checks.append(DoctorCheck("Best PPO model", "OK", str(summary.best_model_path)))
    else:
        checks.append(DoctorCheck("Best PPO model", "WARN", "none found"))

    if summary.history:
        checks.append(DoctorCheck("PPO training history", "OK", f"runs={len(summary.history)}"))
    else:
        checks.append(DoctorCheck("PPO training history", "WARN", "none found"))

    if summary.evaluation_history:
        latest_eval = summary.latest_eval or {}
        checks.append(
            DoctorCheck(
                "PPO eval history",
                "OK",
                f"runs={len(summary.evaluation_history)} mean_reward={latest_eval.get('mean_reward', 'none')}",
            )
        )
    else:
        checks.append(DoctorCheck("PPO eval history", "WARN", "none found"))

    checks.append(DoctorCheck("Observation size", "OK", str(OBSERVATION_SIZE)))
    return DoctorReport(checks=checks)


def format_doctor_report(report: DoctorReport) -> str:
    lines = ["doctor: PPO environment check"]
    for check in report.checks:
        lines.append(f"- {check.name}: {check.status} ({check.detail})")
    return "\n".join(lines)
