from __future__ import annotations

import platform
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .dqn_agent import DEFAULT_DQN_ARTIFACTS_DIR, summarize_dqn
from .observations import OBSERVATION_SIZE


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


def run_doctor(output_dir: str | Path | None = None) -> DoctorReport:
    checks: list[DoctorCheck] = [
        DoctorCheck("Python", "OK", platform.python_version()),
        DoctorCheck("Platform", "OK", platform.platform()),
    ]
    for module_name in ("numpy", "cv2", "gymnasium", "stable_baselines3", "sb3_contrib", "torch", "selenium"):
        ok, detail = _check_import(module_name)
        checks.append(DoctorCheck(module_name, "OK" if ok else "ERROR", detail))
    checks.append(_check_torch_cuda())

    artifacts_root = Path(output_dir) if output_dir is not None else DEFAULT_DQN_ARTIFACTS_DIR
    try:
        artifacts_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=artifacts_root, delete=True):
            pass
        checks.append(DoctorCheck("Artifacts write", "OK", str(artifacts_root)))
    except Exception as exc:
        checks.append(DoctorCheck("Artifacts write", "ERROR", f"{exc.__class__.__name__}: {exc}"))

    summary = summarize_dqn(artifacts_root)
    checks.append(
        DoctorCheck("Best DQN model", "OK", str(summary.best_model_path))
        if summary.best_model_path is not None
        else DoctorCheck("Best DQN model", "WARN", "none found")
    )
    checks.append(DoctorCheck("DQN training history", "OK", f"runs={len(summary.history)}") if summary.history else DoctorCheck("DQN training history", "WARN", "none found"))
    checks.append(DoctorCheck("DQN eval history", "OK", f"runs={len(summary.evaluation_history)}") if summary.evaluation_history else DoctorCheck("DQN eval history", "WARN", "none found"))
    checks.append(DoctorCheck("Observation size", "OK", str(OBSERVATION_SIZE)))
    return DoctorReport(checks=checks)


def format_doctor_report(report: DoctorReport) -> str:
    lines = ["doctor: DQN environment check"]
    for check in report.checks:
        lines.append(f"- {check.name}: {check.status} ({check.detail})")
    return "\n".join(lines)


def _check_import(module_name: str) -> tuple[bool, str]:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        return True, f"{module_name}={version}"
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def _check_torch_cuda() -> DoctorCheck:
    try:
        import torch
    except Exception as exc:
        return DoctorCheck("Torch CUDA", "WARN", f"torch unavailable: {exc.__class__.__name__}: {exc}")

    cuda_available = bool(torch.cuda.is_available())
    cuda_version = getattr(torch.version, "cuda", None)
    if not cuda_available:
        return DoctorCheck("Torch CUDA", "WARN", f"available=False torch_cuda={cuda_version or 'none'}")
    try:
        device_name = torch.cuda.get_device_name(0)
    except Exception:
        device_name = "unknown"
    return DoctorCheck("Torch CUDA", "OK", f"available=True torch_cuda={cuda_version} device0={device_name}")
