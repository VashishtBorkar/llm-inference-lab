from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def _run_version_command(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output or None


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    commit = _run_version_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status = _run_version_command(["git", "status", "--porcelain"], cwd=repo_root)
    return {
        "commit": commit if commit and not commit.startswith("fatal:") else None,
        "dirty": bool(status) if status is not None else None,
    }


def collect_environment(repo_root: Path) -> dict[str, Any]:
    nvidia = _run_version_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "operating_system": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu": {
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER"),
            "logical_count": os.cpu_count(),
        },
        "gpu": {"nvidia_smi": nvidia},
        "ollama_version": _run_version_command(["ollama", "--version"]),
        "git": _git_metadata(repo_root),
    }
