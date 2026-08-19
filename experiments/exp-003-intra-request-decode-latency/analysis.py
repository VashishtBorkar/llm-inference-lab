from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from inference_lab.analysis import (
    ExperimentDataset,
    load_experiment_dataset,
    write_analysis_manifest,
    write_csv,
)
from inference_lab.experiment import ExperimentError

EXPERIMENT_DIR = Path(__file__).resolve().parent
TARGET_TOKENS = 4096
WINDOW_SIZE = 256
EXPECTED_TRIALS = 5
MAX_START_TEMPERATURE_C = 70.0
MAX_START_UTILIZATION_PCT = 25.0
START_TEMPERATURE_SPREAD_WARNING_C = 2.0

LIMITER_FIELDS = (
    "limited_sw_power",
    "limited_hw_slowdown",
    "limited_sw_thermal",
    "limited_hw_thermal",
    "limited_hw_power_brake",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ExperimentError(f"Missing stream-timing artifact: {path}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExperimentError(
                f"Invalid JSON in {path}:{line_number}: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise ExperimentError(f"Expected an object in {path}:{line_number}")
        records.append(value)
    return records


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _minimum(values: list[float]) -> float | None:
    return min(values) if values else None


def _maximum(values: list[float]) -> float | None:
    return max(values) if values else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _numeric(records: list[dict[str, Any]], field: str) -> list[float]:
    return [
        float(record[field])
        for record in records
        if _number(record.get(field)) is not None
    ]


def _gpu_window(
    telemetry: list[dict[str, Any]], start_ms: float, end_ms: float
) -> dict[str, Any]:
    samples = [
        sample
        for sample in telemetry
        if _number(sample.get("sample_offset_ms")) is not None
        and start_ms <= float(sample["sample_offset_ms"]) <= end_ms
    ]
    temperatures = _numeric(samples, "temperature_c")
    clocks = _numeric(samples, "sm_clock_mhz")
    utilization = _numeric(samples, "gpu_utilization_pct")
    power = _numeric(samples, "power_draw_w")
    result: dict[str, Any] = {
        "gpu_sample_count": len(samples),
        "gpu_temperature_start_c": temperatures[0] if temperatures else None,
        "gpu_temperature_mean_c": _mean(temperatures),
        "gpu_temperature_max_c": _maximum(temperatures),
        "gpu_sm_clock_mean_mhz": _mean(clocks),
        "gpu_sm_clock_min_mhz": _minimum(clocks),
        "gpu_utilization_mean_pct": _mean(utilization),
        "gpu_power_mean_w": _mean(power),
    }
    for field in LIMITER_FIELDS:
        result[f"gpu_{field}_seen"] = any(sample.get(field) is True for sample in samples)
    result["gpu_limiter_seen"] = any(
        result[f"gpu_{field}_seen"] for field in LIMITER_FIELDS
    )
    return result


def _measured_request(run: dict[str, Any]) -> dict[str, Any]:
    measured = [request for request in run["requests"] if not request.get("is_warmup")]
    if len(measured) != 1:
        raise ExperimentError(
            f"Run {run['manifest']['run_id']} must contain exactly one measured request; "
            f"found {len(measured)}"
        )
    request = measured[0]
    problems: list[str] = []
    if request.get("status") != "success":
        problems.append(f"status={request.get('status')}")
    if request.get("output_tokens") != TARGET_TOKENS:
        problems.append(f"output_tokens={request.get('output_tokens')}")
    if request.get("done_reason") != "length":
        problems.append(f"done_reason={request.get('done_reason')}")
    if request.get("stream_token_count_matches_eval_count") is not True:
        problems.append("stream token coverage does not match eval_count")
    if request.get("stream_selected_token_count") != TARGET_TOKENS:
        problems.append(
            f"stream_selected_token_count={request.get('stream_selected_token_count')}"
        )
    if problems:
        raise ExperimentError(
            f"Run {run['manifest']['run_id']} is not a complete comparable trial: "
            + "; ".join(problems)
        )
    return request


def _token_arrivals(
    stream_events: list[dict[str, Any]], request_id: str
) -> tuple[dict[int, float], dict[int, bool], int]:
    events = [
        event
        for event in stream_events
        if event.get("request_id") == request_id and not event.get("is_warmup")
    ]
    events.sort(key=lambda event: int(event["event_index"]))
    token_events = [
        event
        for event in events
        if isinstance(event.get("selected_token_count"), int)
        and not isinstance(event.get("selected_token_count"), bool)
        and int(event["selected_token_count"]) > 0
    ]
    if not token_events:
        raise ExperimentError(f"Request {request_id} has no selected-token stream events")

    arrivals: dict[int, float] = {}
    grouped: dict[int, bool] = {}
    previous_received_ms: float | None = None
    expected_cumulative = 0
    grouped_events = 0

    for event_number, event in enumerate(token_events, start=1):
        token_count = int(event["selected_token_count"])
        cumulative = int(event["cumulative_selected_token_count"])
        received_ms = float(event["received_offset_ms"])
        if cumulative != expected_cumulative + token_count:
            raise ExperimentError(
                f"Request {request_id} has non-contiguous token counts at stream event "
                f"{event.get('event_index')}"
            )
        if previous_received_ms is None:
            if token_count != 1:
                raise ExperimentError(
                    f"Request {request_id} begins with a grouped {token_count}-token "
                    "event, so early inter-token latency cannot be recovered"
                )
            arrivals[1] = received_ms
            grouped[1] = False
        else:
            delta_ms = received_ms - previous_received_ms
            if delta_ms < 0:
                raise ExperimentError(
                    f"Request {request_id} has non-monotonic stream timestamps"
                )
            per_token_ms = delta_ms / token_count
            if token_count > 1:
                grouped_events += 1
            for token_offset in range(1, token_count + 1):
                token_index = expected_cumulative + token_offset
                arrivals[token_index] = previous_received_ms + per_token_ms * token_offset
                grouped[token_index] = token_count > 1
        expected_cumulative = cumulative
        previous_received_ms = received_ms

    if expected_cumulative != TARGET_TOKENS:
        raise ExperimentError(
            f"Request {request_id} stream ends at token {expected_cumulative}; "
            f"expected {TARGET_TOKENS}"
        )
    if set(arrivals) != set(range(1, TARGET_TOKENS + 1)):
        raise ExperimentError(f"Request {request_id} has missing token-position timings")
    return arrivals, grouped, grouped_events


def _start_gate_result(run: dict[str, Any]) -> dict[str, Any]:
    result = run["manifest"].get("telemetry", {}).get("start_gate", {}).get("result")
    if not isinstance(result, dict) or result.get("status") != "satisfied":
        raise ExperimentError(
            f"Run {run['manifest']['run_id']} has no satisfied telemetry start gate"
        )
    temperature = _number(result.get("final_temperature_c"))
    utilization = _number(result.get("final_gpu_utilization_pct"))
    if temperature is None or temperature > MAX_START_TEMPERATURE_C:
        raise ExperimentError(
            f"Run {run['manifest']['run_id']} has invalid gate temperature {temperature}"
        )
    if utilization is None or utilization > MAX_START_UTILIZATION_PCT:
        raise ExperimentError(
            f"Run {run['manifest']['run_id']} has invalid gate utilization {utilization}"
        )
    return result


def _build_trial(
    run: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    request = _measured_request(run)
    stream_events = _load_jsonl(run["directory"] / "stream_events.jsonl")
    arrivals, grouped, grouped_event_count = _token_arrivals(
        stream_events, str(request["request_id"])
    )
    context = run["manifest"]["experiment"]
    trial_number = int(context["trial_number"])
    gate = _start_gate_result(run)
    telemetry = run["telemetry"]

    token_itl = {
        token_index: arrivals[token_index] - arrivals[token_index - 1]
        for token_index in range(2, TARGET_TOKENS + 1)
    }
    window_rows: list[dict[str, Any]] = []
    for window_number, window_start in enumerate(
        range(1, TARGET_TOKENS + 1, WINDOW_SIZE), start=1
    ):
        window_end = min(window_start + WINDOW_SIZE - 1, TARGET_TOKENS)
        interval_start = max(2, window_start)
        intervals = [
            token_itl[token_index]
            for token_index in range(interval_start, window_end + 1)
        ]
        grouped_tokens = sum(
            grouped[token_index]
            for token_index in range(interval_start, window_end + 1)
        )
        start_offset = arrivals[window_start]
        end_offset = arrivals[window_end]
        mean_itl = _mean(intervals)
        row = {
            "execution_id": context["execution_id"],
            "run_id": run["manifest"]["run_id"],
            "request_id": request["request_id"],
            "condition_id": context["condition_id"],
            "trial_number": trial_number,
            "schedule_position": context["schedule_position"],
            "window_number": window_number,
            "token_start": window_start,
            "token_end": window_end,
            "token_midpoint": (window_start + window_end) / 2,
            "inter_token_interval_count": len(intervals),
            "client_itl_mean_ms": mean_itl,
            "client_itl_median_ms": _median(intervals),
            "client_itl_p95_ms": _percentile(intervals, 0.95),
            "client_window_output_tokens_per_second": (
                1000 / mean_itl if mean_itl is not None and mean_itl > 0 else None
            ),
            "cumulative_client_tpot_ms": (
                (arrivals[window_end] - arrivals[1]) / (window_end - 1)
                if window_end > 1
                else None
            ),
            "grouped_token_count": grouped_tokens,
            "grouped_token_fraction": (
                grouped_tokens / len(intervals) if intervals else None
            ),
            "window_started_offset_ms": start_offset,
            "window_completed_offset_ms": end_offset,
            **_gpu_window(telemetry, start_offset, end_offset),
        }
        window_rows.append(row)

    first_quarter = [token_itl[index] for index in range(2, 1025)]
    last_quarter = [token_itl[index] for index in range(3073, 4097)]
    all_itl = list(token_itl.values())
    first_mean = statistics.fmean(first_quarter)
    last_mean = statistics.fmean(last_quarter)
    request_gpu = _gpu_window(
        telemetry,
        float(request["started_offset_ms"]),
        float(request["completed_offset_ms"]),
    )
    eval_duration_ms = _number(request.get("ollama_eval_duration_ms"))
    trial_row = {
        "execution_id": context["execution_id"],
        "run_id": run["manifest"]["run_id"],
        "request_id": request["request_id"],
        "trial_number": trial_number,
        "schedule_position": context["schedule_position"],
        "output_tokens": request["output_tokens"],
        "done_reason": request["done_reason"],
        "response_sha256": request["response_sha256"],
        "stream_event_count": request.get("stream_events_recorded"),
        "stream_grouped_event_count": grouped_event_count,
        "stream_grouped_token_fraction": (
            sum(grouped[index] for index in range(2, TARGET_TOKENS + 1))
            / (TARGET_TOKENS - 1)
        ),
        "client_ttft_ms": request.get("client_ttft_ms"),
        "client_e2e_ms": request.get("client_e2e_ms"),
        "client_full_request_tpot_ms": statistics.fmean(all_itl),
        "ollama_average_decode_ms_per_token": (
            eval_duration_ms / TARGET_TOKENS if eval_duration_ms is not None else None
        ),
        "first_quarter_client_itl_ms": first_mean,
        "last_quarter_client_itl_ms": last_mean,
        "first_to_last_client_itl_change_pct": (
            ((last_mean / first_mean) - 1) * 100 if first_mean > 0 else None
        ),
        "start_gate_temperature_c": gate["final_temperature_c"],
        "start_gate_gpu_utilization_pct": gate["final_gpu_utilization_pct"],
        "start_gate_wait_seconds": gate["wait_seconds"],
        **request_gpu,
    }
    return window_rows, trial_row


def _aggregate_windows(window_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for window_number in range(1, TARGET_TOKENS // WINDOW_SIZE + 1):
        rows = [row for row in window_rows if row["window_number"] == window_number]
        if len(rows) != EXPECTED_TRIALS:
            raise ExperimentError(
                f"Window {window_number} contains {len(rows)} trial estimates; "
                f"expected {EXPECTED_TRIALS}"
            )
        trial_itl = _numeric(rows, "client_itl_mean_ms")
        trial_cumulative = _numeric(rows, "cumulative_client_tpot_ms")
        temperatures = _numeric(rows, "gpu_temperature_mean_c")
        clocks = _numeric(rows, "gpu_sm_clock_mean_mhz")
        aggregates.append(
            {
                "window_number": window_number,
                "token_start": rows[0]["token_start"],
                "token_end": rows[0]["token_end"],
                "token_midpoint": rows[0]["token_midpoint"],
                "trial_count": len(rows),
                "trial_mean_itl_median_ms": _median(trial_itl),
                "trial_mean_itl_mean_ms": _mean(trial_itl),
                "trial_mean_itl_min_ms": _minimum(trial_itl),
                "trial_mean_itl_max_ms": _maximum(trial_itl),
                "trial_mean_itl_stdev_ms": (
                    statistics.stdev(trial_itl) if len(trial_itl) > 1 else 0.0
                ),
                "cumulative_tpot_median_ms": _median(trial_cumulative),
                "gpu_temperature_median_c": _median(temperatures),
                "gpu_sm_clock_median_mhz": _median(clocks),
                "limiter_trial_count": sum(row["gpu_limiter_seen"] for row in rows),
                "grouped_token_fraction_max": _maximum(
                    _numeric(rows, "grouped_token_fraction")
                ),
            }
        )
    return aggregates


def _validate_trial_controls(
    dataset: ExperimentDataset, trial_rows: list[dict[str, Any]]
) -> list[str]:
    if len(dataset.runs) != EXPECTED_TRIALS:
        raise ExperimentError(
            f"Execution contains {len(dataset.runs)} completed runs; "
            f"expected {EXPECTED_TRIALS}"
        )
    digests = {
        run["manifest"].get("engine", {}).get("model_metadata", {}).get("digest")
        for run in dataset.runs
    }
    if len(digests) != 1 or None in digests:
        raise ExperimentError("Trials do not share one recorded model digest")
    gpu_uuids = {
        sample.get("gpu_uuid")
        for run in dataset.runs
        for sample in run["telemetry"]
        if sample.get("gpu_uuid") is not None
    }
    if len(gpu_uuids) != 1:
        raise ExperimentError("Trials do not share one recorded GPU UUID")

    warnings: list[str] = []
    start_temperatures = _numeric(trial_rows, "start_gate_temperature_c")
    spread = max(start_temperatures) - min(start_temperatures)
    if spread > START_TEMPERATURE_SPREAD_WARNING_C:
        warnings.append(
            f"Start-gate release temperatures span {spread:.1f} C, exceeding the "
            f"planned {START_TEMPERATURE_SPREAD_WARNING_C:.1f} C comparability check."
        )
    response_hashes = {row["response_sha256"] for row in trial_rows}
    if len(response_hashes) != 1:
        warnings.append(
            "Deterministic trials produced different response hashes; content-dependent "
            "decode variation remains a possible confounder."
        )
    grouped_fraction = max(
        float(row["stream_grouped_token_fraction"]) for row in trial_rows
    )
    if grouped_fraction > 0:
        warnings.append(
            f"Up to {grouped_fraction:.2%} of token intervals came from grouped stream "
            "events and use within-event equal-spacing estimates."
        )
    return warnings


def _save(figure: Any, figures_dir: Path, stem: str) -> list[str]:
    paths: list[str] = []
    for extension in ("svg", "png"):
        path = figures_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path.relative_to(EXPERIMENT_DIR).as_posix())
    return paths


def _make_figures(
    window_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    figures_dir: Path,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required; install the analysis extra with "
            "`python -m pip install -e .[analysis]`"
        ) from exc

    figures: list[str] = []
    colors = plt.get_cmap("tab10")

    figure, axis = plt.subplots(figsize=(10, 5.5))
    for trial_number in range(1, EXPECTED_TRIALS + 1):
        rows = [row for row in window_rows if row["trial_number"] == trial_number]
        axis.plot(
            [row["token_midpoint"] for row in rows],
            [row["client_itl_mean_ms"] for row in rows],
            marker="o",
            alpha=0.55,
            color=colors(trial_number - 1),
            label=f"Trial {trial_number}",
        )
    x = [row["token_midpoint"] for row in aggregate_rows]
    median = [row["trial_mean_itl_median_ms"] for row in aggregate_rows]
    low = [row["trial_mean_itl_min_ms"] for row in aggregate_rows]
    high = [row["trial_mean_itl_max_ms"] for row in aggregate_rows]
    axis.plot(x, median, color="black", linewidth=2.5, label="Across-trial median")
    axis.fill_between(x, low, high, color="black", alpha=0.12, label="Trial range")
    axis.set_xlabel("Generated token position")
    axis.set_ylabel("Mean client inter-token latency (ms)")
    axis.set_title("Inter-token latency across one 4096-token response")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, fontsize=8)
    figures.extend(_save(figure, figures_dir, "inter-token-latency"))
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for trial_number in range(1, EXPECTED_TRIALS + 1):
        rows = [row for row in window_rows if row["trial_number"] == trial_number]
        color = colors(trial_number - 1)
        position = [row["token_midpoint"] for row in rows]
        axes[0].plot(position, [row["client_itl_mean_ms"] for row in rows], color=color)
        axes[1].plot(position, [row["gpu_temperature_mean_c"] for row in rows], color=color)
        axes[2].plot(position, [row["gpu_sm_clock_mean_mhz"] for row in rows], color=color)
    axes[0].set_ylabel("Client ITL (ms)")
    axes[1].set_ylabel("GPU temperature (C)")
    axes[2].set_ylabel("Mean SM clock (MHz)")
    axes[2].set_xlabel("Generated token position")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("Decode latency and GPU state by token position")
    figures.extend(_save(figure, figures_dir, "decode-gpu-timeline"))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    for trial_number in range(1, EXPECTED_TRIALS + 1):
        rows = [row for row in window_rows if row["trial_number"] == trial_number]
        axis.plot(
            [row["token_end"] for row in rows],
            [row["cumulative_client_tpot_ms"] for row in rows],
            marker="o",
            color=colors(trial_number - 1),
            label=f"Trial {trial_number}",
        )
    axis.set_xlabel("Generated tokens completed")
    axis.set_ylabel("Cumulative client TPOT (ms)")
    axis.set_title("Average streaming pace if the response ended at each checkpoint")
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, fontsize=8)
    figures.extend(_save(figure, figures_dir, "cumulative-tpot"))
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    color_values = [row["token_midpoint"] for row in window_rows]
    scatter = axes[0].scatter(
        [row["gpu_temperature_mean_c"] for row in window_rows],
        [row["client_itl_mean_ms"] for row in window_rows],
        c=color_values,
        cmap="viridis",
    )
    axes[1].scatter(
        [row["gpu_sm_clock_mean_mhz"] for row in window_rows],
        [row["client_itl_mean_ms"] for row in window_rows],
        c=color_values,
        cmap="viridis",
    )
    axes[0].set_xlabel("Mean GPU temperature (C)")
    axes[1].set_xlabel("Mean SM clock (MHz)")
    for axis in axes:
        axis.set_ylabel("Mean client ITL (ms)")
        axis.grid(alpha=0.2)
    figure.colorbar(scatter, ax=axes, label="Generated token position")
    figure.suptitle("Client inter-token latency associations with GPU state")
    figures.extend(_save(figure, figures_dir, "latency-vs-gpu-state"))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    trial_numbers = [int(row["trial_number"]) for row in trial_rows]
    width = 0.36
    axis.bar(
        [trial - width / 2 for trial in trial_numbers],
        [row["first_quarter_client_itl_ms"] for row in trial_rows],
        width=width,
        label="Tokens 2-1024",
    )
    axis.bar(
        [trial + width / 2 for trial in trial_numbers],
        [row["last_quarter_client_itl_ms"] for row in trial_rows],
        width=width,
        label="Tokens 3073-4096",
    )
    axis.set_xticks(trial_numbers)
    axis.set_xlabel("Trial")
    axis.set_ylabel("Mean client inter-token latency (ms)")
    axis.set_title("First versus last response quarter")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figures.extend(_save(figure, figures_dir, "first-vs-last-quarter"))
    plt.close(figure)
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze exp-003 intra-request decode-latency runs."
    )
    parser.add_argument("--execution-id")
    args = parser.parse_args()

    dataset = load_experiment_dataset(EXPERIMENT_DIR, execution_id=args.execution_id)
    window_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for run in sorted(
        dataset.runs,
        key=lambda item: int(item["manifest"]["experiment"]["trial_number"]),
    ):
        run_windows, trial = _build_trial(run)
        window_rows.extend(run_windows)
        trial_rows.append(trial)

    warnings = _validate_trial_controls(dataset, trial_rows)
    aggregate_rows = _aggregate_windows(window_rows)
    results_dir = EXPERIMENT_DIR / "results"
    figures_dir = EXPERIMENT_DIR / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results_dir / "window-measurements.csv", window_rows)
    write_csv(results_dir / "window-aggregate.csv", aggregate_rows)
    write_csv(results_dir / "trial-summary.csv", trial_rows)
    figures = _make_figures(window_rows, aggregate_rows, trial_rows, figures_dir)
    write_analysis_manifest(results_dir / "analysis-manifest.json", dataset, figures)

    manifest_path = results_dir / "analysis-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "target_output_tokens": TARGET_TOKENS,
            "token_window_size": WINDOW_SIZE,
            "trial_count": len(trial_rows),
            "aggregation_unit": "per-trial token-window estimate",
            "warnings": warnings,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    first = statistics.median(
        row["first_quarter_client_itl_ms"] for row in trial_rows
    )
    last = statistics.median(
        row["last_quarter_client_itl_ms"] for row in trial_rows
    )
    print(f"Analyzed execution: {dataset.execution_id}")
    print(f"Trials: {len(trial_rows)}")
    print(f"Median first-quarter client ITL: {first:.3f} ms")
    print(f"Median last-quarter client ITL: {last:.3f} ms")
    for warning in warnings:
        print(f"warning: {warning}")
    print(f"Results: {results_dir}")
    print(f"Figures: {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
