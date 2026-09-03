from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inference_lab.engines.ollama import OllamaAdapter
from inference_lab.models import GenerationObservation, Scenario
from inference_lab.workload import load_workload

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_CONFIG = EXPERIMENT_DIR / "profile.toml"


class ProfileError(RuntimeError):
    """Raised when a profiler capture cannot preserve the experiment controls."""


@dataclass(frozen=True)
class ProfileConfig:
    experiment_id: str
    model: str
    workload_path: Path
    prompt_tokens: tuple[int, ...]
    pilot_prompt_tokens: tuple[int, ...]
    repetitions: int
    base_url: str
    keep_alive: str
    request_timeout_seconds: float
    readiness_timeout_seconds: float
    capture_settle_seconds: float
    shutdown_timeout_seconds: float
    required_gpu_name: str
    ollama_environment: dict[str, str]
    nsys_trace: tuple[str, ...]
    cuda_graph_trace: str
    sample: str
    cpu_context_switches: str
    export: str


def _require_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ProfileError(f"profile configuration requires [{key}]")
    return value


def _positive_int_list(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ProfileError(f"{name} must be a non-empty integer array")
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in value):
        raise ProfileError(f"{name} must contain only positive integers")
    if len(set(value)) != len(value):
        raise ProfileError(f"{name} must not contain duplicates")
    return tuple(value)


def load_config(path: Path) -> ProfileConfig:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if data.get("schema_version") != "1.0":
        raise ProfileError("profile schema_version must be '1.0'")
    profile = _require_table(data, "profile")
    ollama = _require_table(data, "ollama")
    environment = _require_table(ollama, "environment")
    nsys = _require_table(data, "nsight_systems")

    workload_path = (REPO_ROOT / str(profile.get("workload", ""))).resolve()
    if not workload_path.is_relative_to(REPO_ROOT):
        raise ProfileError("profile workload must remain inside the repository")
    repetitions = profile.get("repetitions")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ProfileError("profile.repetitions must be a positive integer")
    trace = nsys.get("trace")
    if not isinstance(trace, list) or not trace or not all(isinstance(x, str) for x in trace):
        raise ProfileError("nsight_systems.trace must be a non-empty string array")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in environment.items()):
        raise ProfileError("ollama.environment values must be strings")

    return ProfileConfig(
        experiment_id=str(profile["id"]),
        model=str(profile["model"]),
        workload_path=workload_path,
        prompt_tokens=_positive_int_list(profile.get("prompt_tokens"), "profile.prompt_tokens"),
        pilot_prompt_tokens=_positive_int_list(
            profile.get("pilot_prompt_tokens"), "profile.pilot_prompt_tokens"
        ),
        repetitions=repetitions,
        base_url=str(profile["base_url"]).rstrip("/"),
        keep_alive=str(profile["keep_alive"]),
        request_timeout_seconds=float(profile["request_timeout_seconds"]),
        readiness_timeout_seconds=float(profile["readiness_timeout_seconds"]),
        capture_settle_seconds=float(profile["capture_settle_seconds"]),
        shutdown_timeout_seconds=float(profile["shutdown_timeout_seconds"]),
        required_gpu_name=str(profile["required_gpu_name"]),
        ollama_environment=dict(environment),
        nsys_trace=tuple(trace),
        cuda_graph_trace=str(nsys["cuda_graph_trace"]),
        sample=str(nsys["sample"]),
        cpu_context_switches=str(nsys["cpu_context_switches"]),
        export=str(nsys["export"]),
    )


def load_scenario_pairs(config: ProfileConfig) -> dict[int, dict[str, Scenario]]:
    workload = load_workload(config.workload_path)
    pairs: dict[int, dict[str, Scenario]] = {}
    for scenario in workload.scenarios:
        expected = scenario.validation.get("expected_prompt_tokens")
        role = scenario.validation.get("profile_role")
        if not isinstance(expected, int) or role not in {"warmup", "profile"}:
            raise ProfileError(
                f"scenario {scenario.scenario_id} lacks profiling token/role metadata"
            )
        by_role = pairs.setdefault(expected, {})
        if role in by_role:
            raise ProfileError(f"duplicate {role} scenario for {expected} tokens")
        by_role[role] = scenario

    for prompt_tokens in config.prompt_tokens:
        roles = pairs.get(prompt_tokens, {})
        if set(roles) != {"warmup", "profile"}:
            raise ProfileError(
                f"{prompt_tokens} tokens requires one warmup and one profile scenario"
            )
    unknown_pilot = set(config.pilot_prompt_tokens) - set(config.prompt_tokens)
    if unknown_pilot:
        raise ProfileError(f"pilot lengths are absent from the matrix: {sorted(unknown_pilot)}")
    return pairs


def build_nsys_command(
    config: ProfileConfig,
    *,
    session_name: str,
    output_prefix: Path,
) -> list[str]:
    return [
        "nsys",
        "profile",
        "--start-later=true",
        f"--session-new={session_name}",
        f"--trace={','.join(config.nsys_trace)}",
        f"--cuda-graph-trace={config.cuda_graph_trace}",
        f"--sample={config.sample}",
        f"--cpuctxsw={config.cpu_context_switches}",
        f"--export={config.export}",
        "--force-overwrite=true",
        "--show-output=true",
        "--kill=none",
        "--wait=all",
        f"--output={output_prefix}",
        "ollama",
        "serve",
    ]


def _endpoint_ready(base_url: str, timeout_seconds: float = 1.0) -> bool:
    request = urllib.request.Request(f"{base_url}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return getattr(response, "status", 200) == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _wait_for_ollama(
    config: ProfileConfig,
    profile_process: subprocess.Popen[bytes],
) -> None:
    deadline = time.monotonic() + config.readiness_timeout_seconds
    while time.monotonic() < deadline:
        return_code = profile_process.poll()
        if return_code is not None:
            raise ProfileError(f"Nsight/Ollama exited before readiness with code {return_code}")
        if _endpoint_ready(config.base_url):
            return
        time.sleep(0.25)
    raise ProfileError("timed out waiting for the profiled Ollama server")


def _version(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return (result.stdout or result.stderr).strip()


def preflight(config: ProfileConfig, *, require_allocation: bool) -> dict[str, Any]:
    missing = [command for command in ("ollama", "nsys", "nvidia-smi") if shutil.which(command) is None]
    if missing:
        raise ProfileError(f"missing required commands: {', '.join(missing)}")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if require_allocation and not slurm_job_id:
        raise ProfileError("profile capture requires an active Slurm GPU allocation")
    if _endpoint_ready(config.base_url):
        raise ProfileError(
            f"Ollama is already listening at {config.base_url}; stop it so Nsight can launch it"
        )

    gpu_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    gpu_names = [line.strip() for line in gpu_result.stdout.splitlines() if line.strip()]
    if require_allocation:
        if gpu_result.returncode != 0 or len(gpu_names) != 1:
            raise ProfileError("the allocation must expose exactly one NVIDIA GPU")
        if gpu_names[0] != config.required_gpu_name:
            raise ProfileError(
                f"expected {config.required_gpu_name}, but the allocation exposes {gpu_names[0]}"
            )

    return {
        "slurm_job_id": slurm_job_id,
        "hostname": os.uname().nodename,
        "gpu_names": gpu_names,
        "ollama_version": _version(["ollama", "--version"]),
        "nsys_version": _version(["nsys", "--version"]),
        "ncu_path": shutil.which("ncu"),
        "ncu_version": _version(["ncu", "--version"]) if shutil.which("ncu") else None,
    }


def _observation_metadata(observation: GenerationObservation) -> dict[str, Any]:
    return {
        "started_at_utc": observation.started_at_utc,
        "started_perf_ns": observation.started_perf_ns,
        "completed_perf_ns": observation.completed_perf_ns,
        "status": observation.status,
        "http_status": observation.http_status,
        "error_type": observation.error_type,
        "error_message": observation.error_message,
        "response_chars": observation.response_chars,
        "response_sha256": observation.response_sha256,
        "done_reason": observation.done_reason,
        "prompt_eval_count": observation.prompt_eval_count,
        "prompt_eval_duration_ns": observation.prompt_eval_duration_ns,
        "eval_count": observation.eval_count,
        "eval_duration_ns": observation.eval_duration_ns,
        "total_duration_ns": observation.total_duration_ns,
        "load_duration_ns": observation.load_duration_ns,
    }


def _validate_observation(
    observation: GenerationObservation,
    *,
    expected_prompt_tokens: int,
    phase: str,
) -> None:
    if observation.status != "success":
        raise ProfileError(f"{phase} request failed: {observation.error_message}")
    if observation.prompt_eval_count != expected_prompt_tokens:
        raise ProfileError(
            f"{phase} expected {expected_prompt_tokens} prompt tokens, "
            f"Ollama reported {observation.prompt_eval_count}"
        )
    if observation.response_text.strip() != "OK":
        raise ProfileError(f"{phase} response did not pass exact-match validation")


def _stop_process_group(process: subprocess.Popen[bytes], timeout_seconds: float) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5.0)


def capture_one(
    config: ProfileConfig,
    *,
    prompt_tokens: int,
    repetition: int,
    scenarios: dict[str, Scenario],
    execution_dir: Path,
    environment_metadata: dict[str, Any],
) -> Path:
    capture_dir = execution_dir / f"prompt-{prompt_tokens:05d}" / f"repeat-{repetition:02d}"
    capture_dir.mkdir(parents=True, exist_ok=False)
    output_prefix = capture_dir / "nsight"
    session_name = f"prefill_{prompt_tokens}_{repetition}_{os.getpid()}"
    command = build_nsys_command(
        config,
        session_name=session_name,
        output_prefix=output_prefix,
    )
    environment = os.environ.copy()
    environment.update(config.ollama_environment)
    log_path = capture_dir / "ollama.log"
    profile_process: subprocess.Popen[bytes] | None = None
    collection_started = False
    warmup_observation: GenerationObservation | None = None
    measured_observation: GenerationObservation | None = None

    with log_path.open("wb") as log_handle:
        try:
            profile_process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _wait_for_ollama(config, profile_process)
            adapter = OllamaAdapter(
                base_url=config.base_url,
                timeout_seconds=config.request_timeout_seconds,
                keep_alive=config.keep_alive,
            )
            adapter.model_metadata(config.model)
            warmup_observation = adapter.generate(
                model=config.model,
                scenario=scenarios["warmup"],
            )
            _validate_observation(
                warmup_observation,
                expected_prompt_tokens=prompt_tokens,
                phase="warmup",
            )

            start_process = subprocess.Popen(
                ["nsys", "start", f"--session={session_name}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            time.sleep(config.capture_settle_seconds)
            if start_process.poll() not in {None, 0}:
                output = start_process.stdout.read().decode("utf-8", errors="replace")
                raise ProfileError(f"nsys start failed: {output.strip()}")
            collection_started = True

            measured_observation = adapter.generate(
                model=config.model,
                scenario=scenarios["profile"],
            )
            _validate_observation(
                measured_observation,
                expected_prompt_tokens=prompt_tokens,
                phase="profile",
            )
            stop_result = subprocess.run(
                ["nsys", "stop", f"--session={session_name}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=60.0,
            )
            collection_started = False
            if stop_result.returncode != 0:
                raise ProfileError(f"nsys stop failed: {stop_result.stdout}{stop_result.stderr}")
            start_process.wait(timeout=60.0)
        finally:
            if collection_started:
                subprocess.run(
                    ["nsys", "stop", f"--session={session_name}"],
                    check=False,
                    capture_output=True,
                    timeout=30.0,
                )
            if profile_process is not None:
                _stop_process_group(profile_process, config.shutdown_timeout_seconds)

    metadata = {
        "schema_version": "1.0",
        "experiment_id": config.experiment_id,
        "prompt_tokens": prompt_tokens,
        "repetition": repetition,
        "session_name": session_name,
        "profiler_command": command,
        "environment": environment_metadata,
        "warmup_scenario_id": scenarios["warmup"].scenario_id,
        "profile_scenario_id": scenarios["profile"].scenario_id,
        "warmup": _observation_metadata(warmup_observation),
        "profile": _observation_metadata(measured_observation),
        "raw_artifacts": {
            "ollama_log": log_path.name,
            "nsight_prefix": output_prefix.name,
        },
    }
    metadata_path = capture_dir / "capture.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return capture_dir


def _execution_directory(config: ProfileConfig) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = (REPO_ROOT / "runs" / config.experiment_id / timestamp).resolve()
    runs_root = (REPO_ROOT / "runs").resolve()
    if not destination.is_relative_to(runs_root):
        raise ProfileError("profile output escaped the ignored runs directory")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture isolated Ollama prefill requests with Nsight Systems."
    )
    parser.add_argument(
        "mode",
        choices=("dry-run", "preflight", "pilot", "all"),
        help="validate locally, inspect the allocation, capture 2K/16K, or capture the matrix",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        nargs="+",
        help="override the configured token matrix for a targeted capture",
    )
    parser.add_argument("--repetitions", type=int, help="override configured repetitions")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    scenarios = load_scenario_pairs(config)
    if args.prompt_tokens:
        prompt_tokens = tuple(args.prompt_tokens)
    elif args.mode == "pilot":
        prompt_tokens = config.pilot_prompt_tokens
    else:
        prompt_tokens = config.prompt_tokens
    unknown = set(prompt_tokens) - set(config.prompt_tokens)
    if unknown:
        raise ProfileError(f"requested lengths are absent from the workload: {sorted(unknown)}")
    repetitions = args.repetitions or config.repetitions
    if repetitions < 1:
        raise ProfileError("repetitions must be positive")

    if args.mode == "dry-run":
        sample_command = build_nsys_command(
            config,
            session_name="prefill_dry_run",
            output_prefix=Path("runs") / config.experiment_id / "DRY_RUN" / "nsight",
        )
        print(
            json.dumps(
                {
                    "experiment_id": config.experiment_id,
                    "prompt_tokens": prompt_tokens,
                    "repetitions": repetitions,
                    "sample_command": sample_command,
                    "scenario_pairs": {
                        str(tokens): {
                            role: scenario.scenario_id for role, scenario in by_role.items()
                        }
                        for tokens, by_role in scenarios.items()
                    },
                },
                indent=2,
            )
        )
        return 0

    environment_metadata = preflight(config, require_allocation=True)
    print(json.dumps(environment_metadata, indent=2))
    if args.mode == "preflight":
        return 0

    execution_dir = _execution_directory(config)
    index = {
        "schema_version": "1.0",
        "experiment_id": config.experiment_id,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "prompt_tokens": list(prompt_tokens),
        "repetitions": repetitions,
        "environment": environment_metadata,
        "captures": [],
    }
    index_path = execution_dir / "profile-index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    for prompt_length in prompt_tokens:
        for repetition in range(1, repetitions + 1):
            print(f"Capturing {prompt_length} prompt tokens, repetition {repetition}")
            capture_dir = capture_one(
                config,
                prompt_tokens=prompt_length,
                repetition=repetition,
                scenarios=scenarios[prompt_length],
                execution_dir=execution_dir,
                environment_metadata=environment_metadata,
            )
            index["captures"].append(capture_dir.relative_to(execution_dir).as_posix())
            index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    index["completed_at_utc"] = datetime.now(UTC).isoformat()
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"Completed profile matrix: {execution_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
