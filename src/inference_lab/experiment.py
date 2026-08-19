from __future__ import annotations

import hashlib
import json
import random
import re
import time
import tomllib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inference_lab.engines.ollama import OllamaAdapter
from inference_lab.models import (
    ExperimentRunContext,
    RunConfig,
    StreamTimingConfig,
    TelemetryConfig,
    TelemetryStartGateConfig,
)
from inference_lab.runner import ProgressCallback, RunResult, run_benchmark
from inference_lab.workload import load_workload


class ExperimentError(ValueError):
    """Raised when an experiment specification is invalid or cannot run."""


RUN_DEFAULT_KEYS = {
    "engine",
    "base_url",
    "model",
    "workload",
    "warmup",
    "warmup_max_output_tokens",
    "repetitions",
    "concurrency",
    "timeout_seconds",
    "keep_alive",
    "capture_output",
    "inter_request_delay_seconds",
}

CONDITION_OVERRIDE_KEYS = {
    "warmup",
    "warmup_max_output_tokens",
    "repetitions",
    "concurrency",
    "timeout_seconds",
    "keep_alive",
    "inter_request_delay_seconds",
}


@dataclass(frozen=True)
class ExperimentCondition:
    condition_id: str
    label: str
    run_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    path: Path
    directory: Path
    repo_root: Path
    schema_version: str
    experiment_id: str
    title: str
    question: str
    hypothesis: str
    specification_sha256: str
    defaults: dict[str, Any]
    telemetry: TelemetryConfig
    stream_timing: StreamTimingConfig
    trials_per_condition: int
    condition_order: str
    order_seed: int
    between_runs_seconds: float
    conditions: tuple[ExperimentCondition, ...]


@dataclass(frozen=True)
class ExperimentExecutionResult:
    experiment_id: str
    execution_id: str
    execution_dir: Path
    index: dict[str, Any]
    runs: tuple[RunResult, ...]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_table(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ExperimentError(f"experiment.toml: [{name}] table is required")
    return value


def _require_string(data: dict[str, Any], name: str, source: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ExperimentError(f"{source}.{name} must be a non-empty string")
    return value.strip()


def _integer(data: dict[str, Any], name: str, default: int, minimum: int) -> int:
    value = data.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ExperimentError(f"{name} must be an integer of at least {minimum}")
    return value


def _number(data: dict[str, Any], name: str, default: float, minimum: float) -> float:
    value = data.get(name, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < minimum:
        raise ExperimentError(f"{name} must be a number of at least {minimum}")
    return float(value)


def resolve_experiment_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_dir():
        resolved = resolved / "experiment.toml"
    if not resolved.is_file():
        raise ExperimentError(f"Experiment specification does not exist: {resolved}")
    return resolved


def _inside_repo(path: Path, repo_root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ExperimentError(f"{label} must stay inside the repository: {resolved}") from exc
    return resolved


def _validate_run_values(values: dict[str, Any], source: str) -> None:
    if values.get("engine") != "ollama":
        raise ExperimentError(f"{source}.engine must currently be 'ollama'")
    for field_name in ("model", "base_url", "keep_alive"):
        if not isinstance(values.get(field_name), str) or not values[field_name].strip():
            raise ExperimentError(f"{source}.{field_name} must be a non-empty string")
    workload = values.get("workload")
    if not isinstance(workload, (str, Path)) or not str(workload).strip():
        raise ExperimentError(f"{source}.workload must be a non-empty path")
    for field_name, minimum in (("warmup", 0), ("repetitions", 1), ("concurrency", 1)):
        value = values.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ExperimentError(
                f"{source}.{field_name} must be an integer of at least {minimum}"
            )
    warmup_max_output_tokens = values.get("warmup_max_output_tokens")
    if warmup_max_output_tokens is not None and (
        not isinstance(warmup_max_output_tokens, int)
        or isinstance(warmup_max_output_tokens, bool)
        or warmup_max_output_tokens < 1
    ):
        raise ExperimentError(
            f"{source}.warmup_max_output_tokens must be a positive integer or omitted"
        )
    for field_name, minimum in (
        ("timeout_seconds", 0.001),
        ("inter_request_delay_seconds", 0.0),
    ):
        value = values.get(field_name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < minimum:
            raise ExperimentError(
                f"{source}.{field_name} must be a number of at least {minimum}"
            )
    if not isinstance(values.get("capture_output"), bool):
        raise ExperimentError(f"{source}.capture_output must be a boolean")
    if values["inter_request_delay_seconds"] > 0 and values["concurrency"] != 1:
        raise ExperimentError(
            f"{source}: inter-request cooldown currently requires concurrency = 1"
        )


def load_experiment(path: Path, *, repo_root: Path) -> ExperimentSpec:
    specification_path = resolve_experiment_path(path)
    _inside_repo(specification_path, repo_root, "experiment specification")
    raw_bytes = specification_path.read_bytes()
    try:
        data = tomllib.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ExperimentError(f"Invalid experiment TOML: {exc}") from exc
    if not isinstance(data, dict):
        raise ExperimentError("experiment.toml must contain a TOML document")

    schema_version = _require_string(data, "schema_version", "experiment.toml")
    if schema_version != "1.0":
        raise ExperimentError(f"Unsupported experiment schema_version: {schema_version}")

    experiment = _require_table(data, "experiment")
    experiment_id = _require_string(experiment, "id", "experiment")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", experiment_id):
        raise ExperimentError("experiment.id must use lowercase letters, digits, and hyphens")
    if specification_path.parent.name != experiment_id:
        raise ExperimentError(
            "experiment.id must match the directory containing experiment.toml"
        )
    title = _require_string(experiment, "title", "experiment")
    question = _require_string(experiment, "question", "experiment")
    hypothesis = _require_string(experiment, "hypothesis", "experiment")

    raw_defaults = _require_table(data, "defaults")
    unknown_defaults = set(raw_defaults) - RUN_DEFAULT_KEYS
    if unknown_defaults:
        raise ExperimentError(
            f"Unknown defaults keys: {', '.join(sorted(unknown_defaults))}"
        )
    defaults = {
        "engine": raw_defaults.get("engine", "ollama"),
        "base_url": raw_defaults.get("base_url", "http://127.0.0.1:11434"),
        "model": raw_defaults.get("model", "qwen3:4b-instruct"),
        "workload": raw_defaults.get("workload"),
        "warmup": raw_defaults.get("warmup", 1),
        "warmup_max_output_tokens": raw_defaults.get("warmup_max_output_tokens"),
        "repetitions": raw_defaults.get("repetitions", 3),
        "concurrency": raw_defaults.get("concurrency", 1),
        "timeout_seconds": raw_defaults.get("timeout_seconds", 300.0),
        "keep_alive": raw_defaults.get("keep_alive", "5m"),
        "capture_output": raw_defaults.get("capture_output", False),
        "inter_request_delay_seconds": raw_defaults.get(
            "inter_request_delay_seconds", 0.0
        ),
    }
    _validate_run_values(defaults, "defaults")
    workload_path = _inside_repo(
        repo_root / str(defaults["workload"]), repo_root, "workload path"
    )
    load_workload(workload_path)
    defaults["workload"] = workload_path

    raw_execution = _require_table(data, "execution")
    trials_per_condition = _integer(
        raw_execution, "trials_per_condition", default=1, minimum=1
    )
    condition_order = raw_execution.get("condition_order", "fixed")
    if condition_order not in {"fixed", "randomized"}:
        raise ExperimentError("execution.condition_order must be 'fixed' or 'randomized'")
    order_seed = _integer(raw_execution, "order_seed", default=0, minimum=0)
    between_runs_seconds = _number(
        raw_execution, "between_runs_seconds", default=0.0, minimum=0.0
    )

    raw_telemetry = _require_table(data, "telemetry")
    raw_start_gate = raw_telemetry.get("start_gate")
    if raw_start_gate is not None and not isinstance(raw_start_gate, dict):
        raise ExperimentError("telemetry.start_gate must be a table")
    start_gate: TelemetryStartGateConfig | None = None
    if isinstance(raw_start_gate, dict):
        unknown_start_gate = set(raw_start_gate) - {
            "max_temperature_c",
            "max_gpu_utilization_pct",
            "consecutive_samples",
            "timeout_seconds",
        }
        if unknown_start_gate:
            raise ExperimentError(
                "Unknown telemetry.start_gate keys: "
                + ", ".join(sorted(unknown_start_gate))
            )
        max_temperature_c = raw_start_gate.get("max_temperature_c")
        if max_temperature_c is not None and (
            not isinstance(max_temperature_c, (int, float))
            or isinstance(max_temperature_c, bool)
            or max_temperature_c <= 0
        ):
            raise ExperimentError(
                "telemetry.start_gate.max_temperature_c must be a positive number"
            )
        max_gpu_utilization_pct = raw_start_gate.get(
            "max_gpu_utilization_pct"
        )
        if max_gpu_utilization_pct is not None and (
            not isinstance(max_gpu_utilization_pct, (int, float))
            or isinstance(max_gpu_utilization_pct, bool)
            or not 0 <= max_gpu_utilization_pct <= 100
        ):
            raise ExperimentError(
                "telemetry.start_gate.max_gpu_utilization_pct must be between 0 and 100"
            )
        if max_temperature_c is None and max_gpu_utilization_pct is None:
            raise ExperimentError(
                "telemetry.start_gate requires at least one threshold"
            )
        start_gate = TelemetryStartGateConfig(
            max_temperature_c=(
                float(max_temperature_c)
                if max_temperature_c is not None
                else None
            ),
            max_gpu_utilization_pct=(
                float(max_gpu_utilization_pct)
                if max_gpu_utilization_pct is not None
                else None
            ),
            consecutive_samples=_integer(
                raw_start_gate, "consecutive_samples", 3, 1
            ),
            timeout_seconds=_number(
                raw_start_gate, "timeout_seconds", 300.0, 0.001
            ),
        )
    telemetry = TelemetryConfig(
        enabled=bool(raw_telemetry.get("enabled", False)),
        required=bool(raw_telemetry.get("required", False)),
        interval_ms=_integer(raw_telemetry, "interval_ms", 500, 100),
        pre_roll_seconds=_number(raw_telemetry, "pre_roll_seconds", 0.0, 0.0),
        post_roll_seconds=_number(raw_telemetry, "post_roll_seconds", 0.0, 0.0),
        start_gate=start_gate,
    )
    for boolean_name in ("enabled", "required"):
        raw_value = raw_telemetry.get(boolean_name, False)
        if not isinstance(raw_value, bool):
            raise ExperimentError(f"telemetry.{boolean_name} must be a boolean")
    if telemetry.required and not telemetry.enabled:
        raise ExperimentError("required telemetry must be enabled")
    if telemetry.start_gate is not None and not telemetry.enabled:
        raise ExperimentError("telemetry start gate requires telemetry to be enabled")

    raw_stream_timing = data.get("stream_timing", {})
    if not isinstance(raw_stream_timing, dict):
        raise ExperimentError("stream_timing must be a table")
    unknown_stream_timing = set(raw_stream_timing) - {
        "enabled",
        "request_token_logprobs",
        "require_token_counts",
        "include_warmup",
    }
    if unknown_stream_timing:
        raise ExperimentError(
            "Unknown stream_timing keys: "
            + ", ".join(sorted(unknown_stream_timing))
        )
    for boolean_name, default in (
        ("enabled", False),
        ("request_token_logprobs", True),
        ("require_token_counts", False),
        ("include_warmup", False),
    ):
        if not isinstance(raw_stream_timing.get(boolean_name, default), bool):
            raise ExperimentError(f"stream_timing.{boolean_name} must be a boolean")
    stream_timing = StreamTimingConfig(
        enabled=raw_stream_timing.get("enabled", False),
        request_token_logprobs=raw_stream_timing.get(
            "request_token_logprobs", True
        ),
        require_token_counts=raw_stream_timing.get("require_token_counts", False),
        include_warmup=raw_stream_timing.get("include_warmup", False),
    )
    if stream_timing.require_token_counts and not stream_timing.enabled:
        raise ExperimentError(
            "required stream token counts need stream timing enabled"
        )
    if (
        stream_timing.require_token_counts
        and not stream_timing.request_token_logprobs
    ):
        raise ExperimentError(
            "required stream token counts need token logprobs requested"
        )
    if stream_timing.require_token_counts and not stream_timing.include_warmup:
        raise ExperimentError(
            "required stream token counts must include warmup so coverage is "
            "checked before measurement"
        )

    raw_conditions = data.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ExperimentError("experiment.toml must define at least one [[conditions]]")
    conditions: list[ExperimentCondition] = []
    seen_ids: set[str] = set()
    for index, raw_condition in enumerate(raw_conditions, start=1):
        source = f"conditions[{index}]"
        if not isinstance(raw_condition, dict):
            raise ExperimentError(f"{source} must be a table")
        condition_id = _require_string(raw_condition, "id", source)
        if condition_id in seen_ids:
            raise ExperimentError(f"Duplicate condition id: {condition_id}")
        seen_ids.add(condition_id)
        label = _require_string(raw_condition, "label", source)
        overrides = raw_condition.get("run", {})
        if not isinstance(overrides, dict):
            raise ExperimentError(f"{source}.run must be a table")
        unknown_overrides = set(overrides) - CONDITION_OVERRIDE_KEYS
        if unknown_overrides:
            raise ExperimentError(
                f"Unknown {source}.run keys: {', '.join(sorted(unknown_overrides))}"
            )
        merged = {**defaults, **overrides}
        _validate_run_values(merged, source)
        if stream_timing.require_token_counts and merged["warmup"] < 1:
            raise ExperimentError(
                f"{source}: required stream token counts need at least one "
                "warmup request"
            )
        conditions.append(
            ExperimentCondition(
                condition_id=condition_id,
                label=label,
                run_overrides=dict(overrides),
            )
        )

    return ExperimentSpec(
        path=specification_path,
        directory=specification_path.parent,
        repo_root=repo_root.resolve(),
        schema_version=schema_version,
        experiment_id=experiment_id,
        title=title,
        question=question,
        hypothesis=hypothesis,
        specification_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        defaults=defaults,
        telemetry=telemetry,
        stream_timing=stream_timing,
        trials_per_condition=trials_per_condition,
        condition_order=condition_order,
        order_seed=order_seed,
        between_runs_seconds=between_runs_seconds,
        conditions=tuple(conditions),
    )


def _execution_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def run_experiment(
    spec: ExperimentSpec,
    *,
    progress: ProgressCallback | None = None,
) -> ExperimentExecutionResult:
    execution_id = _execution_id()
    execution_dir = (
        spec.repo_root / "runs" / spec.experiment_id / execution_id
    ).resolve()
    execution_dir.mkdir(parents=True, exist_ok=False)
    index_path = execution_dir / "experiment-run-index.json"

    schedule = [
        (condition, trial)
        for trial in range(1, spec.trials_per_condition + 1)
        for condition in spec.conditions
    ]
    if spec.condition_order == "randomized":
        random.Random(spec.order_seed).shuffle(schedule)

    index: dict[str, Any] = {
        "index_version": "1.0",
        "experiment_id": spec.experiment_id,
        "experiment_title": spec.title,
        "execution_id": execution_id,
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": None,
        "specification_path": spec.path.relative_to(spec.repo_root).as_posix(),
        "specification_sha256": spec.specification_sha256,
        "condition_order": spec.condition_order,
        "order_seed": spec.order_seed,
        "between_runs_seconds": spec.between_runs_seconds,
        "realized_schedule": [
            {
                "schedule_position": position,
                "condition_id": condition.condition_id,
                "condition_label": condition.label,
                "trial_number": trial,
            }
            for position, (condition, trial) in enumerate(schedule, start=1)
        ],
        "runs": [],
        "failure": None,
    }
    _write_json(index_path, index)
    run_results: list[RunResult] = []

    try:
        for schedule_index, (condition, trial_number) in enumerate(schedule, start=1):
            if schedule_index > 1 and spec.between_runs_seconds:
                time.sleep(spec.between_runs_seconds)

            values = {**spec.defaults, **condition.run_overrides}
            context = ExperimentRunContext(
                experiment_id=spec.experiment_id,
                execution_id=execution_id,
                condition_id=condition.condition_id,
                condition_label=condition.label,
                trial_number=trial_number,
                schedule_position=schedule_index,
                specification_sha256=spec.specification_sha256,
                changed_parameters=dict(condition.run_overrides),
            )
            config = RunConfig(
                model=str(values["model"]),
                workload_path=Path(values["workload"]),
                output_root=execution_dir,
                base_url=str(values["base_url"]),
                warmup=int(values["warmup"]),
                warmup_max_output_tokens=(
                    int(values["warmup_max_output_tokens"])
                    if values["warmup_max_output_tokens"] is not None
                    else None
                ),
                repetitions=int(values["repetitions"]),
                concurrency=int(values["concurrency"]),
                timeout_seconds=float(values["timeout_seconds"]),
                keep_alive=str(values["keep_alive"]),
                capture_output=bool(values["capture_output"]),
                label=(
                    f"{spec.experiment_id}-{condition.condition_id}-"
                    f"trial-{trial_number:02d}"
                ),
                inter_request_delay_seconds=float(
                    values["inter_request_delay_seconds"]
                ),
                telemetry=spec.telemetry,
                stream_timing=spec.stream_timing,
                experiment=context,
            )
            adapter = OllamaAdapter(
                base_url=config.base_url,
                timeout_seconds=config.timeout_seconds,
            )
            run_entry: dict[str, Any] = {
                "schedule_position": schedule_index,
                "condition_id": condition.condition_id,
                "condition_label": condition.label,
                "trial_number": trial_number,
                "run_id": None,
                "run_directory": None,
                "status": "running",
            }
            index["runs"].append(run_entry)
            _write_json(index_path, index)
            directories_before = {
                path.name for path in execution_dir.iterdir() if path.is_dir()
            }
            try:
                result = run_benchmark(
                    config=config,
                    adapter=adapter,
                    repo_root=spec.repo_root,
                    progress=progress,
                )
            except BaseException:
                new_directories = sorted(
                    path
                    for path in execution_dir.iterdir()
                    if path.is_dir() and path.name not in directories_before
                )
                if len(new_directories) == 1:
                    failed_dir = new_directories[0]
                    run_entry["run_id"] = failed_dir.name
                    run_entry["run_directory"] = failed_dir.name
                    try:
                        failed_manifest = json.loads(
                            (failed_dir / "manifest.json").read_text(encoding="utf-8")
                        )
                    except (FileNotFoundError, json.JSONDecodeError):
                        run_entry["status"] = "failed"
                    else:
                        run_entry["status"] = failed_manifest.get("status", "failed")
                else:
                    run_entry["status"] = "failed"
                _write_json(index_path, index)
                raise
            run_results.append(result)
            run_entry.update(
                run_id=result.run_id,
                run_directory=result.run_dir.relative_to(execution_dir).as_posix(),
                status=result.manifest["status"],
            )
            _write_json(index_path, index)
        index["status"] = "completed"
    except BaseException as exc:
        index["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        index["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        index["completed_at_utc"] = datetime.now(UTC).isoformat()
        _write_json(index_path, index)

    return ExperimentExecutionResult(
        experiment_id=spec.experiment_id,
        execution_id=execution_id,
        execution_dir=execution_dir,
        index=index,
        runs=tuple(run_results),
    )
