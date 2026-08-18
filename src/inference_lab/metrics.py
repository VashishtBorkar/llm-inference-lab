from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Iterable
from typing import Any

from inference_lab.models import GenerationObservation, RequestRecord


def ns_to_ms(value: int | None) -> float | None:
    if value is None:
        return None
    return value / 1_000_000


def per_second(count: int | None, duration_ns: int | None) -> float | None:
    if count is None or duration_ns is None or duration_ns <= 0:
        return None
    return count / (duration_ns / 1_000_000_000)


def derive_client_metrics(observation: GenerationObservation) -> dict[str, float | None]:
    first_chunk_ms = None
    if observation.first_chunk_perf_ns is not None:
        first_chunk_ms = ns_to_ms(
            observation.first_chunk_perf_ns - observation.started_perf_ns
        )

    ttft_ms = None
    decode_ms = None
    tpot_ms = None
    decode_tokens_per_second = None
    if observation.first_content_perf_ns is not None:
        ttft_ms = ns_to_ms(
            observation.first_content_perf_ns - observation.started_perf_ns
        )
        decode_duration_ns = (
            observation.completed_perf_ns - observation.first_content_perf_ns
        )
        decode_ms = ns_to_ms(decode_duration_ns)
        if observation.eval_count is not None and observation.eval_count > 1:
            tpot_ms = decode_ms / (observation.eval_count - 1)
            decode_tokens_per_second = per_second(
                observation.eval_count - 1, decode_duration_ns
            )

    return {
        "client_time_to_first_chunk_ms": first_chunk_ms,
        "client_ttft_ms": ttft_ms,
        "client_e2e_ms": ns_to_ms(
            observation.completed_perf_ns - observation.started_perf_ns
        ),
        "client_decode_ms": decode_ms,
        "client_tpot_ms": tpot_ms,
        "client_decode_tokens_per_second": decode_tokens_per_second,
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def metric_stats(values: Iterable[float | None]) -> dict[str, float | int | None]:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
            "stdev": None,
        }
    return {
        "count": len(usable),
        "mean": statistics.fmean(usable),
        "median": statistics.median(usable),
        "p95": percentile(usable, 0.95) if len(usable) >= 20 else None,
        "min": min(usable),
        "max": max(usable),
        "stdev": statistics.stdev(usable) if len(usable) >= 2 else 0.0,
    }


SUMMARY_METRICS = (
    "client_ttft_ms",
    "client_e2e_ms",
    "client_decode_ms",
    "client_tpot_ms",
    "client_decode_tokens_per_second",
    "ollama_prompt_tokens_per_second",
    "ollama_output_tokens_per_second",
    "prompt_tokens",
    "output_tokens",
)


def _summarize_group(records: list[RequestRecord]) -> dict[str, Any]:
    successful = [record for record in records if record.status == "success"]
    quality_passed = [record for record in successful if record.quality_passed]
    return {
        "requests": len(records),
        "successful": len(successful),
        "quality_passed": len(quality_passed),
        "status_counts": dict(Counter(record.status for record in records)),
        "metrics": {
            metric: metric_stats(getattr(record, metric) for record in successful)
            for metric in SUMMARY_METRICS
        },
    }


def summarize_records(
    records: list[RequestRecord], measurement_elapsed_seconds: float
) -> dict[str, Any]:
    measured = [record for record in records if not record.is_warmup]
    overall = _summarize_group(measured)
    successful = overall["successful"]
    overall["measurement_elapsed_seconds"] = measurement_elapsed_seconds
    overall["request_throughput_per_second"] = (
        successful / measurement_elapsed_seconds
        if measurement_elapsed_seconds > 0
        else None
    )

    scenario_ids = sorted({record.scenario_id for record in measured})
    return {
        "summary_version": "1.0",
        "overall": overall,
        "by_scenario": {
            scenario_id: _summarize_group(
                [record for record in measured if record.scenario_id == scenario_id]
            )
            for scenario_id in scenario_ids
        },
    }
