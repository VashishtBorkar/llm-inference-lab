from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference_lab.experiment import ExperimentError, ExperimentSpec, load_experiment


@dataclass(frozen=True)
class ExperimentDataset:
    spec: ExperimentSpec
    execution_id: str
    execution_dir: Path
    index: dict[str, Any]
    runs: tuple[dict[str, Any], ...]
    measurements: tuple[dict[str, Any], ...]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Cannot read analysis artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentError(f"Expected a JSON object in {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ExperimentError(f"Missing analysis artifact: {path}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExperimentError(f"Invalid JSON in {path}:{line_number}: {exc.msg}") from exc
        if isinstance(value, dict):
            records.append(value)
    return records


def _repo_root(experiment_directory: Path) -> Path:
    for candidate in (experiment_directory, *experiment_directory.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ExperimentError("Could not locate repository root from experiment directory")


def _select_execution(
    spec: ExperimentSpec, execution_id: str | None
) -> tuple[Path, dict[str, Any]]:
    experiment_runs = spec.repo_root / "runs" / spec.experiment_id
    if execution_id is not None:
        candidates = [experiment_runs / execution_id]
    else:
        candidates = sorted(
            (path for path in experiment_runs.glob("*") if path.is_dir()),
            reverse=True,
        )
    for directory in candidates:
        index_path = directory / "experiment-run-index.json"
        if not index_path.is_file():
            continue
        index = _load_json(index_path)
        if index.get("status") != "completed":
            continue
        if index.get("specification_sha256") != spec.specification_sha256:
            if execution_id is not None:
                raise ExperimentError(
                    "Requested execution was produced by a different experiment specification"
                )
            continue
        return directory, index
    qualifier = execution_id or "latest complete"
    raise ExperimentError(
        f"No {qualifier} execution matches specification {spec.specification_sha256}"
    )


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _minimum(values: list[float]) -> float | None:
    return min(values) if values else None


def _maximum(values: list[float]) -> float | None:
    return max(values) if values else None


def _numeric(samples: list[dict[str, Any]], field: str) -> list[float]:
    return [
        float(sample[field])
        for sample in samples
        if isinstance(sample.get(field), (int, float))
        and not isinstance(sample.get(field), bool)
    ]


def _join_request_gpu(
    request: dict[str, Any], samples: list[dict[str, Any]]
) -> dict[str, Any]:
    start = float(request["started_offset_ms"])
    end = float(request["completed_offset_ms"])
    overlapping = [
        sample
        for sample in samples
        if isinstance(sample.get("sample_offset_ms"), (int, float))
        and start <= float(sample["sample_offset_ms"]) <= end
    ]
    temperatures = _numeric(overlapping, "temperature_c")
    clocks = _numeric(overlapping, "sm_clock_mhz")
    utilization = _numeric(overlapping, "gpu_utilization_pct")
    power = _numeric(overlapping, "power_draw_w")
    limiter_fields = (
        "limited_sw_power",
        "limited_hw_slowdown",
        "limited_sw_thermal",
        "limited_hw_thermal",
        "limited_hw_power_brake",
    )
    limiter_seen = {
        f"gpu_{field}_seen": any(sample.get(field) is True for sample in overlapping)
        for field in limiter_fields
    }
    return {
        "gpu_sample_count": len(overlapping),
        "gpu_temperature_start_c": temperatures[0] if temperatures else None,
        "gpu_temperature_mean_c": _mean(temperatures),
        "gpu_temperature_max_c": _maximum(temperatures),
        "gpu_sm_clock_mean_mhz": _mean(clocks),
        "gpu_sm_clock_min_mhz": _minimum(clocks),
        "gpu_utilization_mean_pct": _mean(utilization),
        "gpu_power_mean_w": _mean(power),
        "gpu_limiter_seen": any(
            sample.get(field) is True
            for sample in overlapping
            for field in limiter_fields
        ),
        **limiter_seen,
    }


def load_experiment_dataset(
    experiment_directory: Path, *, execution_id: str | None = None
) -> ExperimentDataset:
    directory = experiment_directory.resolve()
    repo_root = _repo_root(directory)
    spec = load_experiment(directory, repo_root=repo_root)
    execution_dir, index = _select_execution(spec, execution_id)
    loaded_runs: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []

    for indexed_run in index.get("runs", []):
        if not isinstance(indexed_run, dict) or indexed_run.get("status") != "completed":
            continue
        run_dir = execution_dir / str(indexed_run["run_directory"])
        manifest = _load_json(run_dir / "manifest.json")
        summary = _load_json(run_dir / "summary.json")
        context = manifest.get("experiment", {})
        if context.get("specification_sha256") != spec.specification_sha256:
            raise ExperimentError(f"Run {run_dir.name} has an incompatible specification")
        requests = _load_jsonl(run_dir / "requests.jsonl")
        telemetry = _load_jsonl(run_dir / "gpu_telemetry.jsonl")
        events = _load_jsonl(run_dir / "events.jsonl")
        measured = [request for request in requests if not request.get("is_warmup")]
        measured.sort(key=lambda item: int(item["sequence_number"]))
        for request_number, request in enumerate(measured, start=1):
            joined = {
                "execution_id": index["execution_id"],
                "run_id": manifest["run_id"],
                "condition_id": context["condition_id"],
                "condition_label": context["condition_label"],
                "trial_number": context["trial_number"],
                "schedule_position": context["schedule_position"],
                "request_number": request_number,
                "request_id": request["request_id"],
                "status": request["status"],
                "quality_passed": request["quality_passed"],
                "started_offset_ms": request["started_offset_ms"],
                "completed_offset_ms": request["completed_offset_ms"],
                "client_ttft_ms": request.get("client_ttft_ms"),
                "client_e2e_ms": request.get("client_e2e_ms"),
                "client_tpot_ms": request.get("client_tpot_ms"),
                "prompt_tokens": request.get("prompt_tokens"),
                "output_tokens": request.get("output_tokens"),
                "ollama_prompt_eval_duration_ms": request.get(
                    "ollama_prompt_eval_duration_ms"
                ),
                "ollama_eval_duration_ms": request.get("ollama_eval_duration_ms"),
                "ollama_output_tokens_per_second": request.get(
                    "ollama_output_tokens_per_second"
                ),
                "done_reason": request.get("done_reason"),
                **_join_request_gpu(request, telemetry),
            }
            measurements.append(joined)
        loaded_runs.append(
            {
                "directory": run_dir,
                "manifest": manifest,
                "summary": summary,
                "requests": requests,
                "telemetry": telemetry,
                "events": events,
            }
        )

    if not loaded_runs:
        raise ExperimentError("Execution contains no completed runs")
    return ExperimentDataset(
        spec=spec,
        execution_id=str(index["execution_id"]),
        execution_dir=execution_dir,
        index=index,
        runs=tuple(loaded_runs),
        measurements=tuple(measurements),
    )


def aggregate_conditions(dataset: ExperimentDataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in dataset.spec.conditions:
        records = [
            record
            for record in dataset.measurements
            if record["condition_id"] == condition.condition_id
            and record["status"] == "success"
        ]
        throughputs = _numeric(records, "ollama_output_tokens_per_second")
        temperatures = _numeric(records, "gpu_temperature_mean_c")
        clocks = _numeric(records, "gpu_sm_clock_mean_mhz")
        powers = _numeric(records, "gpu_power_mean_w")
        segment_size = max(1, math.ceil(len(throughputs) / 4)) if throughputs else 0
        first = _mean(throughputs[:segment_size]) if segment_size else None
        last = _mean(throughputs[-segment_size:]) if segment_size else None
        run = next(
            (
                item
                for item in dataset.runs
                if item["manifest"].get("experiment", {}).get("condition_id")
                == condition.condition_id
            ),
            None,
        )
        timing = run["summary"].get("timing", {}) if run else {}
        measurement_samples = (
            [sample for sample in run["telemetry"] if sample.get("phase") == "measurement"]
            if run
            else []
        )
        sample_temperatures = _numeric(measurement_samples, "temperature_c")

        measurement_sample_count = len(measurement_samples)
        limiter_sample_counts = {
            field: sum(sample.get(field) is True for sample in measurement_samples)
            for field in (
                "limited_sw_power",
                "limited_sw_thermal",
                "limited_hw_slowdown",
                "limited_hw_thermal",
                "limited_hw_power_brake",
            )
        }
        rows.append(
            {
                "condition_id": condition.condition_id,
                "condition_label": condition.label,
                "request_count": len(records),
                "median_output_tokens_per_second": (
                    statistics.median(throughputs) if throughputs else None
                ),
                "mean_output_tokens_per_second": _mean(throughputs),
                "first_segment_output_tokens_per_second": first,
                "last_segment_output_tokens_per_second": last,
                "first_to_last_change_pct": (
                    ((last / first) - 1) * 100
                    if first is not None and last is not None and first != 0
                    else None
                ),
                "mean_gpu_temperature_c": _mean(temperatures),
                "max_gpu_temperature_c": _maximum(temperatures),
                "mean_gpu_sm_clock_mhz": _mean(clocks),
                "mean_gpu_power_w": _mean(powers),
                "limiter_request_count": sum(
                    record["gpu_limiter_seen"] is True for record in records
                ),
                "measurement_gpu_sample_count": measurement_sample_count,
                "max_gpu_temperature_sample_c": _maximum(sample_temperatures),
                "sw_power_limited_sample_count": limiter_sample_counts[
                    "limited_sw_power"
                ],
                "sw_thermal_limited_sample_count": limiter_sample_counts[
                    "limited_sw_thermal"
                ],
                "hw_slowdown_sample_count": limiter_sample_counts[
                    "limited_hw_slowdown"
                ],
                "hw_thermal_limited_sample_count": limiter_sample_counts[
                    "limited_hw_thermal"
                ],
                "hw_power_brake_sample_count": limiter_sample_counts[
                    "limited_hw_power_brake"
                ],
                "measurement_wall_seconds": timing.get("measurement_wall_seconds"),
                "active_request_seconds": timing.get("active_request_seconds"),
                "scheduled_idle_seconds": timing.get("scheduled_idle_seconds"),
                "wall_output_tokens_per_second": timing.get(
                    "wall_output_tokens_per_second"
                ),
                "engine_active_output_tokens_per_second": timing.get(
                    "engine_active_output_tokens_per_second"
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def write_analysis_manifest(
    path: Path, dataset: ExperimentDataset, figures: list[str]
) -> None:
    value = {
        "analysis_version": "1.0",
        "experiment_id": dataset.spec.experiment_id,
        "execution_id": dataset.execution_id,
        "specification_sha256": dataset.spec.specification_sha256,
        "run_ids": [run["manifest"]["run_id"] for run in dataset.runs],
        "measurement_rows": len(dataset.measurements),
        "figures": figures,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
