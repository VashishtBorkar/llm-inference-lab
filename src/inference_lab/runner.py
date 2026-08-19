from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inference_lab import __version__
from inference_lab.engines.base import EngineAdapter
from inference_lab.environment import collect_environment
from inference_lab.metrics import (
    derive_client_metrics,
    ns_to_ms,
    per_second,
    summarize_records,
)
from inference_lab.models import RequestRecord, RunConfig, Scenario
from inference_lab.telemetry import EventTimeline, NvidiaSmiCollector, TelemetryError
from inference_lab.validators import validate_response, validators_passed
from inference_lab.workload import load_workload

ProgressCallback = Callable[[str, RequestRecord], None]


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    records: tuple[RequestRecord, ...]


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: RequestRecord) -> None:
        line = json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with self._lock, self.path.open(
            "a", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(line)
            handle.write("\n")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_label(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return normalized[:48] or "run"


def _make_run_id(model: str, label: str | None) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = _safe_label(label or model)
    return f"{timestamp}-{suffix}-{uuid.uuid4().hex[:8]}"


def _request_record(
    *,
    run_id: str,
    run_origin_perf_ns: int,
    request_id: str,
    sequence_number: int,
    scenario: Scenario,
    iteration: int,
    is_warmup: bool,
    engine_name: str,
    capture_output: bool,
    observation: Any,
) -> RequestRecord:
    client = derive_client_metrics(observation)
    if observation.status == "success":
        validator_results = validate_response(scenario, observation.response_text)
        quality_passed = validators_passed(validator_results)
    else:
        validator_results = {
            "request_success": {
                "passed": False,
                "error": observation.error_message or "request failed",
            }
        }
        quality_passed = False

    return RequestRecord(
        record_version="1.0",
        run_id=run_id,
        request_id=request_id,
        sequence_number=sequence_number,
        scenario_id=scenario.scenario_id,
        task_type=scenario.task_type,
        workload_class=scenario.workload_class,
        response_format=scenario.response_format,
        generation=dict(scenario.generation),
        tags=list(scenario.tags),
        data_classification=scenario.data_classification,
        iteration=iteration,
        is_warmup=is_warmup,
        engine=engine_name,
        model=observation.model,
        status=observation.status,
        quality_passed=quality_passed,
        validator_results=validator_results,
        error_type=observation.error_type,
        error_message=observation.error_message,
        http_status=observation.http_status,
        started_at_utc=observation.started_at_utc,
        started_offset_ms=(observation.started_perf_ns - run_origin_perf_ns) / 1_000_000,
        completed_offset_ms=(observation.completed_perf_ns - run_origin_perf_ns) / 1_000_000,
        client_time_to_first_chunk_ms=client["client_time_to_first_chunk_ms"],
        client_ttft_ms=client["client_ttft_ms"],
        client_e2e_ms=float(client["client_e2e_ms"] or 0.0),
        client_decode_ms=client["client_decode_ms"],
        client_tpot_ms=client["client_tpot_ms"],
        client_decode_tokens_per_second=client[
            "client_decode_tokens_per_second"
        ],
        stream_chunk_count=observation.stream_chunk_count,
        prompt_tokens=observation.prompt_eval_count,
        output_tokens=observation.eval_count,
        ollama_total_duration_ms=ns_to_ms(observation.total_duration_ns),
        ollama_load_duration_ms=ns_to_ms(observation.load_duration_ns),
        ollama_prompt_eval_duration_ms=ns_to_ms(
            observation.prompt_eval_duration_ns
        ),
        ollama_eval_duration_ms=ns_to_ms(observation.eval_duration_ns),
        ollama_prompt_tokens_per_second=per_second(
            observation.prompt_eval_count, observation.prompt_eval_duration_ns
        ),
        ollama_output_tokens_per_second=per_second(
            observation.eval_count, observation.eval_duration_ns
        ),
        done_reason=observation.done_reason,
        response_chars=observation.response_chars,
        response_sha256=observation.response_sha256,
        response_text=observation.response_text if capture_output else None,
    )


def run_benchmark(
    *,
    config: RunConfig,
    adapter: EngineAdapter,
    repo_root: Path,
    progress: ProgressCallback | None = None,
) -> RunResult:
    if config.warmup < 0:
        raise ValueError("warmup must be zero or greater")
    if config.repetitions < 1:
        raise ValueError("repetitions must be at least one")
    if config.concurrency < 1:
        raise ValueError("concurrency must be at least one")
    if config.inter_request_delay_seconds < 0:
        raise ValueError("inter-request delay must be zero or greater")
    if config.inter_request_delay_seconds > 0 and config.concurrency != 1:
        raise ValueError("inter-request delay currently requires concurrency 1")
    if config.telemetry.required and not config.telemetry.enabled:
        raise ValueError("required telemetry must be enabled")
    if config.telemetry.interval_ms < 100:
        raise ValueError("telemetry interval must be at least 100 ms")
    if config.telemetry.pre_roll_seconds < 0 or config.telemetry.post_roll_seconds < 0:
        raise ValueError("telemetry pre-roll and post-roll must be zero or greater")

    workload = load_workload(config.workload_path)
    model_metadata = adapter.model_metadata(config.model)
    run_id = _make_run_id(config.model, config.label)
    run_dir = config.output_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    requests_path = run_dir / "requests.jsonl"
    writer = JsonlWriter(requests_path)

    run_origin_perf_ns = time.perf_counter_ns()
    started_at_utc = _utc_now()
    events = EventTimeline(
        run_dir / "events.jsonl",
        run_id=run_id,
        origin_perf_ns=run_origin_perf_ns,
    )
    experiment_context = config.experiment
    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "harness_version": __version__,
        "run_id": run_id,
        "label": config.label,
        "status": "running",
        "started_at_utc": started_at_utc,
        "completed_at_utc": None,
        "engine": {
            "name": adapter.name,
            "base_url": config.base_url,
            "model": config.model,
            "model_metadata": model_metadata,
            "keep_alive": config.keep_alive,
        },
        "workload": {
            "path": config.workload_path.as_posix(),
            "content_sha256": workload.content_sha256,
            "manifest": workload.manifest,
            "scenario_ids": [scenario.scenario_id for scenario in workload.scenarios],
        },
        "run_configuration": {
            "warmup_per_scenario": config.warmup,
            "repetitions_per_scenario": config.repetitions,
            "concurrency": config.concurrency,
            "traffic_model": "closed_loop",
            "timeout_seconds": config.timeout_seconds,
            "capture_output": config.capture_output,
            "inter_request_delay_seconds": config.inter_request_delay_seconds,
            "telemetry": {
                "enabled": config.telemetry.enabled,
                "required": config.telemetry.required,
                "interval_ms": config.telemetry.interval_ms,
                "pre_roll_seconds": config.telemetry.pre_roll_seconds,
                "post_roll_seconds": config.telemetry.post_roll_seconds,
            },
        },
        "privacy": {
            "raw_prompts_in_results": False,
            "raw_outputs_in_results": config.capture_output,
        },
        "measurement_notes": {
            "ttft": "client request start to first non-empty content or thinking stream event",
            "stream_chunks": "NDJSON response events; not assumed to be token-aligned",
            "ollama_durations": "engine-reported nanosecond fields converted to milliseconds in request records",
            "summary_excludes_warmup": True,
        },
        "environment": collect_environment(repo_root),
    }
    if experiment_context is not None:
        manifest["experiment"] = {
            "experiment_id": experiment_context.experiment_id,
            "execution_id": experiment_context.execution_id,
            "condition_id": experiment_context.condition_id,
            "condition_label": experiment_context.condition_label,
            "trial_number": experiment_context.trial_number,
            "schedule_position": experiment_context.schedule_position,
            "specification_sha256": experiment_context.specification_sha256,
            "changed_parameters": experiment_context.changed_parameters,
        }
    _write_json(run_dir / "manifest.json", manifest)

    records: list[RequestRecord] = []
    records_lock = threading.Lock()
    next_sequence = 0
    collector: NvidiaSmiCollector | None = None
    telemetry_start_error: str | None = None
    summary: dict[str, Any] | None = None
    scheduled_idle_seconds = 0.0
    measurement_elapsed_seconds = 0.0
    final_status = "running"
    failure: dict[str, str] | None = None

    def execute(
        scenario: Scenario,
        iteration: int,
        is_warmup: bool,
        sequence_number: int,
    ) -> RequestRecord:
        request_id = str(uuid.uuid4())
        events.emit(
            "request_start",
            phase="warmup" if is_warmup else "measurement",
            request_id=request_id,
            scenario_id=scenario.scenario_id,
            iteration=iteration,
            is_warmup=is_warmup,
            sequence_number=sequence_number,
        )
        observation = adapter.generate(
            model=config.model,
            scenario=scenario,
            keep_alive=config.keep_alive,
        )
        record = _request_record(
            run_id=run_id,
            run_origin_perf_ns=run_origin_perf_ns,
            request_id=request_id,
            sequence_number=sequence_number,
            scenario=scenario,
            iteration=iteration,
            is_warmup=is_warmup,
            engine_name=adapter.name,
            capture_output=config.capture_output,
            observation=observation,
        )
        writer.write(record)
        events.emit(
            "request_end",
            phase="warmup" if is_warmup else "measurement",
            request_id=request_id,
            scenario_id=scenario.scenario_id,
            iteration=iteration,
            is_warmup=is_warmup,
            sequence_number=sequence_number,
            status=record.status,
        )
        with records_lock:
            records.append(record)
        if progress:
            progress("warmup" if is_warmup else "measured", record)
        return record

    try:
        events.emit("run_start", phase="initializing")
        if config.telemetry.enabled:
            collector = NvidiaSmiCollector(
                run_dir / "gpu_telemetry.jsonl",
                run_id=run_id,
                origin_perf_ns=run_origin_perf_ns,
                interval_ms=config.telemetry.interval_ms,
            )
            try:
                collector.start()
                sample_timeout = max(5.0, config.telemetry.interval_ms / 1000 * 4)
                if not collector.wait_for_sample(sample_timeout):
                    raise TelemetryError("nvidia-smi produced no valid telemetry samples")
            except TelemetryError as exc:
                telemetry_start_error = str(exc)
                if collector is not None:
                    collector.stop()
                    collector = None
                if config.telemetry.required:
                    raise

        if collector is not None:
            collector.set_phase("pre_roll")
        events.emit(
            "phase_start",
            phase="pre_roll",
            duration_seconds=config.telemetry.pre_roll_seconds,
        )
        if config.telemetry.pre_roll_seconds:
            time.sleep(config.telemetry.pre_roll_seconds)
        events.emit("phase_end", phase="pre_roll")

        if collector is not None:
            collector.set_phase("warmup")
        events.emit("phase_start", phase="warmup")
        for warmup_iteration in range(1, config.warmup + 1):
            for scenario in workload.scenarios:
                execute(scenario, warmup_iteration, True, next_sequence)
                next_sequence += 1
        events.emit("phase_end", phase="warmup")

        measured_tasks: list[tuple[Scenario, int, int]] = []
        for iteration in range(1, config.repetitions + 1):
            for scenario in workload.scenarios:
                measured_tasks.append((scenario, iteration, next_sequence))
                next_sequence += 1

        if collector is not None:
            collector.set_phase("measurement")
        events.emit("phase_start", phase="measurement")
        measurement_started_perf_ns = time.perf_counter_ns()
        if config.concurrency == 1:
            for task_index, (scenario, iteration, sequence_number) in enumerate(
                measured_tasks
            ):
                execute(scenario, iteration, False, sequence_number)
                if (
                    config.inter_request_delay_seconds > 0
                    and task_index < len(measured_tasks) - 1
                ):
                    scheduled_idle_seconds += config.inter_request_delay_seconds
                    if collector is not None:
                        collector.set_phase("cooldown")
                    events.emit(
                        "cooldown_start",
                        phase="cooldown",
                        duration_seconds=config.inter_request_delay_seconds,
                        after_sequence_number=sequence_number,
                    )
                    time.sleep(config.inter_request_delay_seconds)
                    events.emit(
                        "cooldown_end",
                        phase="cooldown",
                        after_sequence_number=sequence_number,
                    )
                    if collector is not None:
                        collector.set_phase("measurement")
        else:
            with ThreadPoolExecutor(
                max_workers=config.concurrency,
                thread_name_prefix="inference-lab",
            ) as executor:
                futures = [
                    executor.submit(execute, scenario, iteration, False, sequence_number)
                    for scenario, iteration, sequence_number in measured_tasks
                ]
                for future in as_completed(futures):
                    future.result()
        measurement_completed_perf_ns = time.perf_counter_ns()
        measurement_elapsed_seconds = (
            measurement_completed_perf_ns - measurement_started_perf_ns
        ) / 1_000_000_000
        events.emit("phase_end", phase="measurement")

        records.sort(key=lambda item: item.sequence_number)
        summary = summarize_records(records, measurement_elapsed_seconds)
        measured_records = [record for record in records if not record.is_warmup]
        active_request_seconds = sum(
            record.client_e2e_ms for record in measured_records
        ) / 1000
        generated_tokens = sum(record.output_tokens or 0 for record in measured_records)
        engine_decode_seconds = sum(
            record.ollama_eval_duration_ms or 0 for record in measured_records
        ) / 1000
        summary.update(
            {
                "run_id": run_id,
                "engine": adapter.name,
                "model": config.model,
                "workload_sha256": workload.content_sha256,
                "timing": {
                    "measurement_wall_seconds": measurement_elapsed_seconds,
                    "active_request_seconds": active_request_seconds,
                    "scheduled_idle_seconds": scheduled_idle_seconds,
                    "engine_decode_seconds": engine_decode_seconds,
                    "wall_output_tokens_per_second": (
                        generated_tokens / measurement_elapsed_seconds
                        if measurement_elapsed_seconds > 0
                        else None
                    ),
                    "engine_active_output_tokens_per_second": (
                        generated_tokens / engine_decode_seconds
                        if engine_decode_seconds > 0
                        else None
                    ),
                },
            }
        )
        _write_json(run_dir / "summary.json", summary)

        if collector is not None:
            collector.set_phase("post_roll")
        events.emit(
            "phase_start",
            phase="post_roll",
            duration_seconds=config.telemetry.post_roll_seconds,
        )
        if config.telemetry.post_roll_seconds:
            time.sleep(config.telemetry.post_roll_seconds)
        events.emit("phase_end", phase="post_roll")
        final_status = "completed"
    except BaseException as exc:
        final_status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        failure = {"type": type(exc).__name__, "message": str(exc)}
        events.emit("run_error", phase="failed", **failure)
        raise
    finally:
        if collector is not None:
            collector.set_phase("stopping")
            collector.stop()
        events.emit("run_end", phase=final_status, status=final_status)
        manifest["status"] = final_status
        manifest["completed_at_utc"] = _utc_now()
        manifest["failure"] = failure
        manifest["telemetry"] = (
            collector.metadata()
            if collector is not None
            else {
                "provider": "nvidia-smi",
                "sample_count": 0,
                "start_error": telemetry_start_error,
            }
        )
        artifacts = {"events": "events.jsonl"}
        if requests_path.exists():
            artifacts["requests"] = "requests.jsonl"
        if (run_dir / "summary.json").exists():
            artifacts["summary"] = "summary.json"
        if (run_dir / "gpu_telemetry.jsonl").exists():
            artifacts["gpu_telemetry"] = "gpu_telemetry.jsonl"
        manifest["artifacts"] = artifacts
        manifest["result_counts"] = {
            "warmup": sum(record.is_warmup for record in records),
            "measured": sum(not record.is_warmup for record in records),
            "failed": sum(
                not record.is_warmup and record.status != "success"
                for record in records
            ),
            "quality_failed": sum(
                not record.is_warmup
                and record.status == "success"
                and not record.quality_passed
                for record in records
            ),
        }
        _write_json(run_dir / "manifest.json", manifest)

    if summary is None:
        raise RuntimeError("benchmark completed without a summary")
    return RunResult(
        run_id=run_id,
        run_dir=run_dir,
        manifest=manifest,
        summary=summary,
        records=tuple(records),
    )
