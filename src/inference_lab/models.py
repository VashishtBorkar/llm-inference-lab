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


@dataclass(frozen=True)
class RunConfig:
    model: str
    workload_path: Path
    output_root: Path
    base_url: str = "http://127.0.0.1:11434"
    warmup: int = 1
    repetitions: int = 3
    concurrency: int = 1
    timeout_seconds: float = 300.0
    keep_alive: str = "5m"
    capture_output: bool = False
    label: str | None = None


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
