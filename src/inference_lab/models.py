from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    task_type: str
    workload_class: str
    messages: tuple[dict[str, str], ...]
    generation: dict[str, Any]
    validators: tuple[str, ...]
    response_format: str = "text"
    response_schema: dict[str, Any] | None = None
    validation: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    data_classification: str = "public"


@dataclass(frozen=True)
class WorkloadBundle:
    root: Path
    manifest: dict[str, Any]
    scenarios: tuple[Scenario, ...]
    content_sha256: str


@dataclass(frozen=True)
class StreamEventObservation:
    """Privacy-safe timing metadata for one Ollama NDJSON stream event.

    ``selected_token_count`` is derived from Ollama's selected-token logprob
    entries when requested. It is deliberately a count: token text, byte
    values, and log probabilities are never retained in this artifact.
    """

    event_index: int
    received_perf_ns: int
    previous_event_delta_ns: int | None
    server_created_at: str | None
    content_chars: int
    thinking_chars: int
    cumulative_content_chars: int
    cumulative_thinking_chars: int
    selected_token_count: int | None
    cumulative_selected_token_count: int
    done: bool


@dataclass(frozen=True)
class StreamTimingConfig:
    enabled: bool = False
    request_token_logprobs: bool = True
    require_token_counts: bool = False
    include_warmup: bool = False


@dataclass(frozen=True)
class GenerationObservation:
    started_at_utc: str
    started_perf_ns: int
    first_chunk_perf_ns: int | None
    first_content_perf_ns: int | None
    completed_perf_ns: int
    status: str
    http_status: int | None
    error_type: str | None
    error_message: str | None
    response_text: str
    response_chars: int
    response_sha256: str
    stream_chunk_count: int
    model: str
    done_reason: str | None
    total_duration_ns: int | None
    load_duration_ns: int | None
    prompt_eval_count: int | None
    prompt_eval_duration_ns: int | None
    eval_count: int | None
    eval_duration_ns: int | None
    stream_events: tuple[StreamEventObservation, ...] = ()
    stream_logprobs_requested: bool = False


@dataclass(frozen=True)
class TelemetryStartGateConfig:
    max_temperature_c: float | None = None
    max_gpu_utilization_pct: float | None = None
    consecutive_samples: int = 3
    timeout_seconds: float = 300.0


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool = False
    required: bool = False
    interval_ms: int = 500
    pre_roll_seconds: float = 0.0
    post_roll_seconds: float = 0.0
    start_gate: TelemetryStartGateConfig | None = None


@dataclass(frozen=True)
class ExperimentRunContext:
    experiment_id: str
    execution_id: str
    condition_id: str
    condition_label: str
    trial_number: int
    schedule_position: int
    specification_sha256: str
    changed_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    model: str
    workload_path: Path
    output_root: Path
    base_url: str = "http://127.0.0.1:11434"
    warmup: int = 1
    warmup_max_output_tokens: int | None = None
    repetitions: int = 3
    concurrency: int = 1
    timeout_seconds: float = 300.0
    keep_alive: str = "5m"
    capture_output: bool = False
    label: str | None = None
    inter_request_delay_seconds: float = 0.0
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    stream_timing: StreamTimingConfig = field(default_factory=StreamTimingConfig)
    experiment: ExperimentRunContext | None = None


@dataclass(frozen=True)
class RequestRecord:
    record_version: str
    run_id: str
    request_id: str
    sequence_number: int
    scenario_id: str
    task_type: str
    workload_class: str
    response_format: str
    generation: dict[str, Any]
    tags: list[str]
    data_classification: str
    iteration: int
    is_warmup: bool
    engine: str
    model: str
    status: str
    quality_passed: bool
    validator_results: dict[str, dict[str, Any]]
    error_type: str | None
    error_message: str | None
    http_status: int | None
    started_at_utc: str
    started_offset_ms: float
    completed_offset_ms: float
    client_time_to_first_chunk_ms: float | None
    client_ttft_ms: float | None
    client_e2e_ms: float
    client_decode_ms: float | None
    client_tpot_ms: float | None
    client_decode_tokens_per_second: float | None
    stream_chunk_count: int
    prompt_tokens: int | None
    output_tokens: int | None
    ollama_total_duration_ms: float | None
    ollama_load_duration_ms: float | None
    ollama_prompt_eval_duration_ms: float | None
    ollama_eval_duration_ms: float | None
    ollama_prompt_tokens_per_second: float | None
    ollama_output_tokens_per_second: float | None
    done_reason: str | None
    response_chars: int
    response_sha256: str
    response_text: str | None
    stream_timing_enabled: bool = False
    stream_events_recorded: int = 0
    stream_events_with_token_counts: int = 0
    stream_one_token_events: int = 0
    stream_grouped_token_events: int = 0
    stream_selected_token_count: int = 0
    stream_max_tokens_per_event: int | None = None
    stream_token_coverage_ratio: float | None = None
    stream_token_count_matches_eval_count: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
