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
TARGET_PROMPT_TOKENS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
EXPECTED_TRIALS = 3
EXPECTED_REPETITIONS = 7
EXPECTED_SAMPLES_PER_LENGTH = EXPECTED_TRIALS * EXPECTED_REPETITIONS
CONTEXT_WINDOW = 32768
MAX_OUTPUT_TOKENS = 8


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _numeric(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [
        float(row[field])
        for row in rows
        if _number(row.get(field)) is not None
    ]


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _request_gpu(
    telemetry: list[dict[str, Any]], start_ms: float, end_ms: float
) -> dict[str, Any]:
    samples = [
        sample
        for sample in telemetry
        if _number(sample.get("sample_offset_ms")) is not None
        and start_ms <= float(sample["sample_offset_ms"]) <= end_ms
    ]
    memory = _numeric(samples, "memory_used_mib")
    utilization = _numeric(samples, "gpu_utilization_pct")
    temperatures = _numeric(samples, "temperature_c")
    clocks = _numeric(samples, "sm_clock_mhz")
    power = _numeric(samples, "power_draw_w")
    return {
        "gpu_sample_count": len(samples),
        "gpu_memory_used_max_mib": max(memory) if memory else None,
        "gpu_utilization_mean_pct": _mean(utilization),
        "gpu_utilization_max_pct": max(utilization) if utilization else None,
        "gpu_temperature_mean_c": _mean(temperatures),
        "gpu_temperature_max_c": max(temperatures) if temperatures else None,
        "gpu_sm_clock_mean_mhz": _mean(clocks),
        "gpu_power_mean_w": _mean(power),
    }


def _target_from_scenario(scenario_id: str) -> int:
    if not scenario_id.startswith("prompt-"):
        raise ExperimentError(f"Unexpected scenario ID: {scenario_id}")
    try:
        target = int(scenario_id.removeprefix("prompt-"))
    except ValueError as exc:
        raise ExperimentError(f"Invalid prompt-length scenario ID: {scenario_id}") from exc
    if target not in TARGET_PROMPT_TOKENS:
        raise ExperimentError(f"Unexpected target prompt length: {target}")
    return target


def _request_rows(dataset: ExperimentDataset) -> list[dict[str, Any]]:
    if len(dataset.runs) != EXPECTED_TRIALS:
        raise ExperimentError(
            f"Execution contains {len(dataset.runs)} runs; expected {EXPECTED_TRIALS}"
        )

    rows: list[dict[str, Any]] = []
    for run in dataset.runs:
        context = run["manifest"]["experiment"]
        trial_number = int(context["trial_number"])
        for request in run["requests"]:
            if request.get("is_warmup"):
                continue
            scenario_id = str(request["scenario_id"])
            target_tokens = _target_from_scenario(scenario_id)
            problems: list[str] = []
            if request.get("status") != "success":
                problems.append(f"status={request.get('status')}")
            if request.get("quality_passed") is not True:
                problems.append("quality validation failed")
            prompt_tokens = _number(request.get("prompt_tokens"))
            prompt_duration = _number(
                request.get("engine_prompt_eval_duration_ms")
            )
            prompt_throughput = _number(
                request.get("engine_prompt_tokens_per_second")
            )
            if prompt_tokens is None or prompt_tokens <= 0:
                problems.append("missing prompt token count")
            if prompt_duration is None or prompt_duration <= 0:
                problems.append("missing prompt-evaluation duration")
            if prompt_throughput is None or prompt_throughput <= 0:
                problems.append("missing prompt throughput")
            generation = request.get("generation", {})
            if generation.get("context_window") != CONTEXT_WINDOW:
                problems.append(
                    f"context_window={generation.get('context_window')}"
                )
            if generation.get("max_output_tokens") != MAX_OUTPUT_TOKENS:
                problems.append(
                    f"max_output_tokens={generation.get('max_output_tokens')}"
                )
            if problems:
                raise ExperimentError(
                    f"Request {request['request_id']} is not comparable: "
                    + "; ".join(problems)
                )

            start_ms = float(request["started_offset_ms"])
            end_ms = float(request["completed_offset_ms"])
            rows.append(
                {
                    "execution_id": context["execution_id"],
                    "run_id": run["manifest"]["run_id"],
                    "request_id": request["request_id"],
                    "trial_number": trial_number,
                    "iteration": request["iteration"],
                    "sequence_number": request["sequence_number"],
                    "scenario_id": scenario_id,
                    "target_prompt_tokens": target_tokens,
                    "actual_prompt_tokens": int(prompt_tokens),
                    "prompt_token_error": int(prompt_tokens) - target_tokens,
                    "prompt_eval_duration_ms": prompt_duration,
                    "prompt_tokens_per_second": prompt_throughput,
                    "client_ttft_ms": request.get("client_ttft_ms"),
                    "client_e2e_ms": request.get("client_e2e_ms"),
                    "output_tokens": request.get("output_tokens"),
                    "done_reason": request.get("done_reason"),
                    "engine_load_duration_ms": request.get(
                        "engine_load_duration_ms"
                    ),
                    **_request_gpu(run["telemetry"], start_ms, end_ms),
                }
            )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for target in TARGET_PROMPT_TOKENS:
        selected = [row for row in rows if row["target_prompt_tokens"] == target]
        if len(selected) != EXPECTED_SAMPLES_PER_LENGTH:
            raise ExperimentError(
                f"Prompt length {target} has {len(selected)} samples; "
                f"expected {EXPECTED_SAMPLES_PER_LENGTH}"
            )
        actual = _numeric(selected, "actual_prompt_tokens")
        durations = _numeric(selected, "prompt_eval_duration_ms")
        throughputs = _numeric(selected, "prompt_tokens_per_second")
        ttft = _numeric(selected, "client_ttft_ms")
        e2e = _numeric(selected, "client_e2e_ms")
        memory = _numeric(selected, "gpu_memory_used_max_mib")
        aggregates.append(
            {
                "target_prompt_tokens": target,
                "sample_count": len(selected),
                "actual_prompt_tokens_median": _median(actual),
                "actual_prompt_tokens_min": min(actual),
                "actual_prompt_tokens_max": max(actual),
                "prompt_eval_duration_median_ms": _median(durations),
                "prompt_eval_duration_mean_ms": _mean(durations),
                "prompt_eval_duration_p95_ms": _percentile(durations, 0.95),
                "prompt_eval_duration_min_ms": min(durations),
                "prompt_eval_duration_max_ms": max(durations),
                "prompt_eval_duration_stdev_ms": statistics.stdev(durations),
                "prompt_throughput_median_tokens_per_second": _median(throughputs),
                "prompt_throughput_mean_tokens_per_second": _mean(throughputs),
                "prompt_throughput_p95_tokens_per_second": _percentile(
                    throughputs, 0.95
                ),
                "client_ttft_median_ms": _median(ttft),
                "client_ttft_p95_ms": _percentile(ttft, 0.95),
                "client_e2e_median_ms": _median(e2e),
                "gpu_memory_used_max_mib": max(memory) if memory else None,
                "requests_with_gpu_samples": sum(
                    int(row["gpu_sample_count"]) > 0 for row in selected
                ),
            }
        )
    return aggregates


def _validate_controls(
    dataset: ExperimentDataset, rows: list[dict[str, Any]]
) -> list[str]:
    digests = {
        run["manifest"].get("engine", {}).get("model_metadata", {}).get("digest")
        for run in dataset.runs
    }
    if len(digests) != 1 or None in digests:
        raise ExperimentError("Runs do not share one recorded Ollama model digest")
    gpu_uuids = {
        sample.get("gpu_uuid")
        for run in dataset.runs
        for sample in run["telemetry"]
        if sample.get("gpu_uuid") is not None
    }
    if len(gpu_uuids) != 1:
        raise ExperimentError("Runs do not share one recorded GPU UUID")

    warnings: list[str] = []
    token_errors = [int(row["prompt_token_error"]) for row in rows]
    if any(token_errors):
        warnings.append(
            "Ollama-reported prompt counts differ from tokenizer targets by "
            f"{min(token_errors)} to {max(token_errors)} tokens; plots use actual counts."
        )
    missing_gpu = sum(int(row["gpu_sample_count"]) == 0 for row in rows)
    if missing_gpu:
        warnings.append(
            f"{missing_gpu} short requests completed between 100 ms GPU samples; "
            "request-level GPU fields are incomplete for those rows."
        )
    return warnings


def _normalize_generated_text(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )


def _save(figure: Any, figures_dir: Path, stem: str) -> list[str]:
    paths: list[str] = []
    for extension in ("svg", "png"):
        path = figures_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        if extension == "svg":
            _normalize_generated_text(path)
        paths.append(path.relative_to(EXPERIMENT_DIR).as_posix())
    return paths


def _make_figures(
    aggregates: list[dict[str, Any]], figures_dir: Path
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required; install with `python -m pip install -e .[analysis]`"
        ) from exc

    x = [float(row["actual_prompt_tokens_median"]) for row in aggregates]
    labels = [str(row["target_prompt_tokens"]) for row in aggregates]
    figures: list[str] = []

    figure, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    duration = [float(row["prompt_eval_duration_median_ms"]) for row in aggregates]
    duration_low = [float(row["prompt_eval_duration_min_ms"]) for row in aggregates]
    duration_high = [float(row["prompt_eval_duration_max_ms"]) for row in aggregates]
    throughput = [
        float(row["prompt_throughput_median_tokens_per_second"])
        for row in aggregates
    ]
    axes[0].plot(x, duration, marker="o", linewidth=2, label="Median")
    axes[0].fill_between(
        x, duration_low, duration_high, alpha=0.18, label="Observed range"
    )
    axes[0].set_ylabel("Prompt evaluation (ms)")
    axes[0].set_title("Ollama prefill duration by actual prompt length")
    axes[0].legend()
    axes[1].plot(x, throughput, marker="o", linewidth=2, color="tab:orange")
    axes[1].set_ylabel("Prompt tokens/s")
    axes[1].set_xlabel("Actual prompt tokens reported by Ollama (log2 scale)")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(x, labels)
        axis.grid(alpha=0.2)
    figures.extend(_save(figure, figures_dir, "prefill-scaling"))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(
        x,
        [float(row["client_ttft_median_ms"]) for row in aggregates],
        marker="o",
        linewidth=2,
        label="Client TTFT",
    )
    axis.plot(
        x,
        [float(row["client_e2e_median_ms"]) for row in aggregates],
        marker="o",
        linewidth=2,
        label="End-to-end latency",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(x, labels)
    axis.set_xlabel("Actual prompt tokens reported by Ollama")
    axis.set_ylabel("Latency (ms)")
    axis.set_title("Client latency by prompt length")
    axis.grid(alpha=0.2)
    axis.legend()
    figures.extend(_save(figure, figures_dir, "client-latency-scaling"))
    plt.close(figure)
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze exp-005 natural-corpus prompt-length prefill scaling."
    )
    parser.add_argument("--execution-id")
    args = parser.parse_args()

    dataset = load_experiment_dataset(EXPERIMENT_DIR, execution_id=args.execution_id)
    rows = _request_rows(dataset)
    warnings = _validate_controls(dataset, rows)
    aggregates = _aggregate(rows)

    results_dir = EXPERIMENT_DIR / "results"
    figures_dir = EXPERIMENT_DIR / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    measurement_path = results_dir / "request-measurements.csv"
    aggregate_path = results_dir / "prompt-length-aggregate.csv"
    write_csv(measurement_path, rows)
    write_csv(aggregate_path, aggregates)
    _normalize_generated_text(measurement_path)
    _normalize_generated_text(aggregate_path)
    figures = _make_figures(aggregates, figures_dir)
    manifest_path = results_dir / "analysis-manifest.json"
    write_analysis_manifest(manifest_path, dataset, figures)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "aggregation_unit": "individual measured request",
            "target_prompt_tokens": list(TARGET_PROMPT_TOKENS),
            "samples_per_prompt_length": EXPECTED_SAMPLES_PER_LENGTH,
            "trial_count": EXPECTED_TRIALS,
            "repetitions_per_trial": EXPECTED_REPETITIONS,
            "warnings": warnings,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Analyzed execution: {dataset.execution_id}")
    print(f"Measured requests: {len(rows)}")
    for row in aggregates:
        print(
            f"{row['target_prompt_tokens']:>5} target tokens: "
            f"{row['prompt_eval_duration_median_ms']:.2f} ms median prefill, "
            f"{row['prompt_throughput_median_tokens_per_second']:.2f} tokens/s"
        )
    for warning in warnings:
        print(f"warning: {warning}")
    print(f"Results: {results_dir}")
    print(f"Figures: {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
