from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
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
from inference_lab.models import (
    RequestRecord,
    RunConfig,
    Scenario,
    StreamEventObservation,
)
from inference_lab.telemetry import EventTimeline, NvidiaSmiCollector, TelemetryError
from inference_lab.validators import validate_response, validators_passed
from inference_lab.workload import load_workload

ProgressCallback = Callable[[str, RequestRecord], None]


class StreamTimingError(RuntimeError):
    """Raised when required selected-token timing coverage is unavailable."""


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


class JsonObjectJsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.count = 0

    def write(self, value: dict[str, Any]) -> None:
        line = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self.path.open(
            "a", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(line)
            handle.write("\n")
            self.count += 1


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


def _stream_timing_metrics(observation: Any, *, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {
            "stream_timing_enabled": False,
            "stream_events_recorded": 0,
            "stream_events_with_token_counts": 0,
            "stream_one_token_events": 0,
            "stream_grouped_token_events": 0,
            "stream_selected_token_count": 0,
            "stream_max_tokens_per_event": None,
            "stream_token_coverage_ratio": None,
            "stream_token_count_matches_eval_count": None,
        }

    events = observation.stream_events
    counts = [
        event.selected_token_count
        for event in events
        if event.selected_token_count is not None
    ]
    selected_token_count = sum(counts)
    eval_count = observation.eval_count
    coverage_ratio = (
        selected_token_count / eval_count
        if isinstance(eval_count, int) and eval_count > 0
        else None
    )
    matches = (
        selected_token_count == eval_count
        if isinstance(eval_count, int) and eval_count >= 0
        else None
    )
    return {
        "stream_timing_enabled": True,
        "stream_events_recorded": len(events),
        "stream_events_with_token_counts": len(counts),
        "stream_one_token_events": sum(count == 1 for count in counts),
        "stream_grouped_token_events": sum(count > 1 for count in counts),
        "stream_selected_token_count": selected_token_count,
        "stream_max_tokens_per_event": max(counts) if counts else None,
        "stream_token_coverage_ratio": coverage_ratio,
        "stream_token_count_matches_eval_count": matches,
    }


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
    stream_timing_enabled: bool,
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

    stream_metrics = _stream_timing_metrics(
        observation, enabled=stream_timing_enabled
    )
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
        **stream_metrics,
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
    if config.warmup_max_output_tokens is not None and (
        not isinstance(config.warmup_max_output_tokens, int)
        or isinstance(config.warmup_max_output_tokens, bool)
        or config.warmup_max_output_tokens < 1
    ):
        raise ValueError("warmup maximum output tokens must be at least one")
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
    start_gate = config.telemetry.start_gate
    if start_gate is not None:
        if not config.telemetry.enabled:
            raise ValueError("telemetry start gate requires telemetry to be enabled")
        if (
            start_gate.max_temperature_c is None
            and start_gate.max_gpu_utilization_pct is None
        ):
            raise ValueError("telemetry start gate requires at least one threshold")
        if (
            start_gate.max_temperature_c is not None
            and start_gate.max_temperature_c <= 0
        ):
            raise ValueError("start-gate maximum temperature must be positive")
        if (
            start_gate.max_gpu_utilization_pct is not None
            and not 0 <= start_gate.max_gpu_utilization_pct <= 100
        ):
            raise ValueError("start-gate maximum GPU utilization must be 0 to 100")
        if start_gate.consecutive_samples < 1:
            raise ValueError("start-gate consecutive samples must be at least one")
        if start_gate.timeout_seconds <= 0:
            raise ValueError("start-gate timeout must be positive")
    if (
        config.stream_timing.require_token_counts
        and not config.stream_timing.enabled
    ):
        raise ValueError("required stream token counts need stream timing enabled")
    if (
        config.stream_timing.require_token_counts
        and not config.stream_timing.request_token_logprobs
    ):
        raise ValueError("required stream token counts need token logprobs requested")
    if (
        config.stream_timing.require_token_counts
        and config.warmup < 1
    ):
        raise ValueError(
            "required stream token counts need at least one warmup request so "
            "coverage is checked before measurement"
        )
    if (
        config.stream_timing.require_token_counts
        and not config.stream_timing.include_warmup
    ):
        raise ValueError(
            "required stream token counts must include warmup so coverage is "
            "checked before measurement"
        )

    workload = load_workload(config.workload_path)
    model_metadata = adapter.model_metadata(config.model)
    run_id = _make_run_id(config.model, config.label)
    run_dir = config.output_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    requests_path = run_dir / "requests.jsonl"
    writer = JsonlWriter(requests_path)
    stream_events_path = run_dir / "stream_events.jsonl"
    stream_writer = (
        JsonObjectJsonlWriter(stream_events_path)
        if config.stream_timing.enabled
        else None
    )

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
            "warmup_max_output_tokens": config.warmup_max_output_tokens,
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
                "start_gate": (
                    {
                        "max_temperature_c": start_gate.max_temperature_c,
                        "max_gpu_utilization_pct": (
                            start_gate.max_gpu_utilization_pct
                        ),
                        "consecutive_samples": start_gate.consecutive_samples,
                        "timeout_seconds": start_gate.timeout_seconds,
                    }
                    if start_gate is not None
                    else None
                ),
            },
            "stream_timing": {
                "enabled": config.stream_timing.enabled,
                "request_token_logprobs": (
                    config.stream_timing.request_token_logprobs
                ),
                "top_logprobs": 0,
                "require_token_counts": config.stream_timing.require_token_counts,
                "include_warmup": config.stream_timing.include_warmup,
            },
        },
        "privacy": {
            "raw_prompts_in_results": False,
            "raw_outputs_in_results": config.capture_output,
        },
        "measurement_notes": {
            "ttft": "client request start to first non-empty content or thinking stream event",
            "stream_chunks": "NDJSON response events; not assumed to be token-aligned",
            "stream_event_timing": (
                "client-observed NDJSON arrival timing; selected-token counts identify "
                "one-token versus grouped events but do not represent GPU-exact token timing"
            ),
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
    telemetry_start_gate_result: dict[str, Any] | None = None
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
        record_stream_timing = config.stream_timing.enabled and (
            not is_warmup or config.stream_timing.include_warmup
        )
        events.emit(
            "request_start",
            phase="warmup" if is_warmup else "measurement",
            request_id=request_id,
            scenario_id=scenario.scenario_id,
            iteration=iteration,
            is_warmup=is_warmup,
            sequence_number=sequence_number,
        )
        generate_options: dict[str, Any] = {}
        if record_stream_timing:
            previous_selected_token_event_perf_ns: int | None = None

            def record_stream_event(event: StreamEventObservation) -> None:
                nonlocal previous_selected_token_event_perf_ns
                if stream_writer is None:
                    return
                token_count = event.selected_token_count
                if token_count is None:
                    token_count_class = "unavailable"
                elif token_count == 0:
                    token_count_class = "no_selected_token"
                elif token_count == 1:
                    token_count_class = "one_selected_token"
                else:
                    token_count_class = "grouped_selected_tokens"
                previous_token_event_delta_ms = None
                if token_count is not None and token_count > 0:
                    if previous_selected_token_event_perf_ns is not None:
                        previous_token_event_delta_ms = (
                            event.received_perf_ns
                            - previous_selected_token_event_perf_ns
                        ) / 1_000_000
                    previous_selected_token_event_perf_ns = event.received_perf_ns
                stream_writer.write(
                    {
                        "record_version": "1.0",
                        "run_id": run_id,
                        "request_id": request_id,
                        "sequence_number": sequence_number,
                        "scenario_id": scenario.scenario_id,
                        "iteration": iteration,
                        "is_warmup": is_warmup,
                        "event_index": event.event_index,
                        "received_offset_ms": (
                            event.received_perf_ns - run_origin_perf_ns
                        )
                        / 1_000_000,
                        "request_offset_ms": (
                            event.received_perf_ns - request_started_perf_ns
                        )
                        / 1_000_000,
                        "previous_event_delta_ms": (
                            event.previous_event_delta_ns / 1_000_000
                            if event.previous_event_delta_ns is not None
                            else None
                        ),
                        "previous_token_event_delta_ms": (
                            previous_token_event_delta_ms
                        ),
                        "server_created_at": event.server_created_at,
                        "content_chars": event.content_chars,
                        "thinking_chars": event.thinking_chars,
                        "cumulative_content_chars": (
                            event.cumulative_content_chars
                        ),
                        "cumulative_thinking_chars": (
                            event.cumulative_thinking_chars
                        ),
                        "selected_token_count": event.selected_token_count,
                        "cumulative_selected_token_count": (
                            event.cumulative_selected_token_count
                        ),
                        "token_count_class": token_count_class,
                        "done": event.done,
                    }
                )

            request_started_perf_ns = time.perf_counter_ns()
            generate_options = {
                "stream_timing": config.stream_timing,
                "stream_event_callback": record_stream_event,
            }
        observation = adapter.generate(
            model=config.model,
            scenario=scenario,
            keep_alive=config.keep_alive,
            **generate_options,
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
            stream_timing_enabled=record_stream_timing,
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
        if (
            record_stream_timing
            and config.stream_timing.require_token_counts
        ):
            if record.status != "success":
                raise StreamTimingError(
                    "required stream-timing calibration request failed for "
                    f"request {request_id}: {record.error_message or record.status}"
                )
            if record.stream_token_count_matches_eval_count is not True:
                raise StreamTimingError(
                    "selected-token stream coverage did not match Ollama eval_count "
                    f"for request {request_id}: observed "
                    f"{record.stream_selected_token_count}, expected "
                    f"{record.output_tokens}"
                )
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
                warmup_scenario = scenario
                if config.warmup_max_output_tokens is not None:
                    warmup_scenario = replace(
                        scenario,
                        generation={
                            **scenario.generation,
                            "max_output_tokens": config.warmup_max_output_tokens,
                        },
                    )
                execute(warmup_scenario, warmup_iteration, True, next_sequence)
                next_sequence += 1
        events.emit("phase_end", phase="warmup")

        if start_gate is not None:
            if collector is None:
                raise TelemetryError(
                    "telemetry start gate cannot run without an active collector"
                )
            collector.set_phase("start_gate")
            events.emit(
                "phase_start",
                phase="start_gate",
                max_temperature_c=start_gate.max_temperature_c,
                max_gpu_utilization_pct=start_gate.max_gpu_utilization_pct,
                consecutive_samples=start_gate.consecutive_samples,
                timeout_seconds=start_gate.timeout_seconds,
            )
            gate_started_monotonic = time.monotonic()
            try:
                telemetry_start_gate_result = collector.wait_for_start_gate(
                    max_temperature_c=start_gate.max_temperature_c,
                    max_gpu_utilization_pct=start_gate.max_gpu_utilization_pct,
                    consecutive_samples=start_gate.consecutive_samples,
                    timeout_seconds=start_gate.timeout_seconds,
                )
            except TelemetryError as exc:
                telemetry_start_gate_result = {
                    "status": "failed",
                    "wait_seconds": time.monotonic() - gate_started_monotonic,
                    "error": str(exc),
                }
                events.emit(
                    "phase_end",
                    phase="start_gate",
                    **telemetry_start_gate_result,
                )
                raise
            events.emit(
                "phase_end",
                phase="start_gate",
                **telemetry_start_gate_result,
            )

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
        manifest["telemetry"]["start_gate"] = {
            "configured": start_gate is not None,
            "result": telemetry_start_gate_result,
        }
        timed_records = [record for record in records if record.stream_timing_enabled]
        measured_timed_records = [
            record for record in timed_records if not record.is_warmup
        ]
        coverage_values = [
            record.stream_token_coverage_ratio
            for record in timed_records
            if record.stream_token_coverage_ratio is not None
        ]
        manifest["stream_timing"] = {
            "enabled": config.stream_timing.enabled,
            "semantics": "client_observed_ndjson_event_arrival",
            "gpu_exact_token_timing": False,
            "token_count_source": (
                "ollama_selected_token_logprob_entry_count"
                if config.stream_timing.request_token_logprobs
                else None
            ),
            "raw_token_values_recorded": False,
            "event_count": stream_writer.count if stream_writer is not None else 0,
            "timed_requests": len(timed_records),
            "measured_timed_requests": len(measured_timed_records),
            "exact_eval_count_coverage_requests": sum(
                record.stream_token_count_matches_eval_count is True
                for record in timed_records
            ),
            "minimum_token_coverage_ratio": (
                min(coverage_values) if coverage_values else None
            ),
            "maximum_tokens_in_one_event": max(
                (
                    record.stream_max_tokens_per_event or 0
                    for record in timed_records
                ),
                default=0,
            ),
            "one_token_events": sum(
                record.stream_one_token_events for record in timed_records
            ),
            "grouped_token_events": sum(
                record.stream_grouped_token_events for record in timed_records
            ),
        }
        artifacts = {"events": "events.jsonl"}
        if requests_path.exists():
            artifacts["requests"] = "requests.jsonl"
        if (run_dir / "summary.json").exists():
            artifacts["summary"] = "summary.json"
        if (run_dir / "gpu_telemetry.jsonl").exists():
            artifacts["gpu_telemetry"] = "gpu_telemetry.jsonl"
        if stream_events_path.exists():
            artifacts["stream_events"] = "stream_events.jsonl"
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
