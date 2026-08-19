from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from inference_lab.analysis import (
    load_experiment_dataset,
    write_analysis_manifest,
    write_csv,
)
from inference_lab.experiment import ExperimentError

EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPECTED_REQUESTS_PER_TRIAL = 100
START_GATE_TEMPERATURE_C = 70.0
START_GATE_UTILIZATION_PCT = 25.0
START_GATE_SAMPLE_COUNT = 10
MATCHED_START_TOLERANCE_C = 3.0


def _numbers(records: Iterable[dict[str, Any]], field: str) -> list[float]:
    return [
        float(record[field])
        for record in records
        if isinstance(record.get(field), (int, float))
        and not isinstance(record.get(field), bool)
        and math.isfinite(float(record[field]))
    ]


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _minimum(values: list[float]) -> float | None:
    return min(values) if values else None


def _maximum(values: list[float]) -> float | None:
    return max(values) if values else None


def _plot_number(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return math.nan


def _format_number(value: Any, decimals: int = 2) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{decimals}f}"
    return "n/a"


def _percent_change(first: float | None, last: float | None) -> float | None:
    if first in (None, 0) or last is None:
        return None
    return ((last / first) - 1.0) * 100.0


def _rolling_mean(values: list[float], window: int = 5) -> list[float]:
    return [
        statistics.fmean(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def _save(figure: Any, figures_dir: Path, stem: str) -> list[str]:
    paths: list[str] = []
    for extension in ("svg", "png"):
        path = figures_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path.relative_to(EXPERIMENT_DIR).as_posix())
    return paths


def _run_trial(run: dict[str, Any]) -> int:
    return int(run["manifest"]["experiment"]["trial_number"])


def _prepare_measurements(
    source: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    rows = [dict(record) for record in source]
    rows.sort(key=lambda row: (int(row["trial_number"]), int(row["request_number"])))
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["trial_number"]), []).append(row)
        duration = row.get("ollama_eval_duration_ms")
        tokens = row.get("output_tokens")
        row["ollama_engine_tpot_ms"] = (
            float(duration) / float(tokens)
            if isinstance(duration, (int, float))
            and isinstance(tokens, (int, float))
            and float(tokens) > 0
            else None
        )

    for trial_rows in grouped.values():
        baseline = _mean(
            _numbers(
                [row for row in trial_rows[:5] if row.get("status") == "success"],
                "ollama_output_tokens_per_second",
            )
        )
        for row in trial_rows:
            throughput = row.get("ollama_output_tokens_per_second")
            row["normalized_throughput"] = (
                float(throughput) / baseline
                if baseline
                and isinstance(throughput, (int, float))
                else None
            )
    return rows


def _start_samples(
    run: dict[str, Any], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return the last gate-sized sample window before measured request one."""
    if not records:
        return []
    first_request_offset = float(records[0]["started_offset_ms"])
    eligible = [
        sample
        for sample in run["telemetry"]
        if isinstance(sample.get("sample_offset_ms"), (int, float))
        and float(sample["sample_offset_ms"]) <= first_request_offset
    ]
    eligible.sort(key=lambda sample: float(sample["sample_offset_ms"]))
    return eligible[-START_GATE_SAMPLE_COUNT:]


def _trial_summaries(
    runs: tuple[dict[str, Any], ...], measurements: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for run in sorted(runs, key=_run_trial):
        trial = _run_trial(run)
        trial_records = [
            row
            for row in measurements
            if int(row["trial_number"]) == trial
        ]
        problems = [
            row
            for row in trial_records
            if row.get("status") != "success"
            or row.get("output_tokens") != 256
            or row.get("done_reason") != "length"
        ]
        if len(trial_records) != EXPECTED_REQUESTS_PER_TRIAL or problems:
            raise ExperimentError(
                f"Trial {trial} is not comparable: found {len(trial_records)} "
                f"measured requests and {len(problems)} failed, short, or "
                "non-length-terminated requests"
            )
        records = trial_records
        records.sort(key=lambda row: int(row["request_number"]))
        segment_size = max(1, math.ceil(len(records) * 0.2)) if records else 0
        throughputs = _numbers(records, "ollama_output_tokens_per_second")
        tpots = _numbers(records, "ollama_engine_tpot_ms")
        first_throughput = _mean(
            _numbers(records[:segment_size], "ollama_output_tokens_per_second")
        )
        last_throughput = _mean(
            _numbers(records[-segment_size:], "ollama_output_tokens_per_second")
        )
        first_tpot = _mean(_numbers(records[:segment_size], "ollama_engine_tpot_ms"))
        last_tpot = _mean(_numbers(records[-segment_size:], "ollama_engine_tpot_ms"))
        start_samples = _start_samples(run, records)
        start_temperatures = _numbers(start_samples, "temperature_c")
        start_utilization = _numbers(start_samples, "gpu_utilization_pct")
        measurement_samples = [
            sample
            for sample in run["telemetry"]
            if sample.get("phase") == "measurement"
        ]
        timing = run["summary"].get("timing", {})
        gate_observed = (
            len(start_samples) == START_GATE_SAMPLE_COUNT
            and len(start_temperatures) == START_GATE_SAMPLE_COUNT
            and len(start_utilization) == START_GATE_SAMPLE_COUNT
            and max(start_temperatures) <= START_GATE_TEMPERATURE_C
            and max(start_utilization) <= START_GATE_UTILIZATION_PCT
        )
        summaries.append(
            {
                "trial_number": trial,
                "run_id": run["manifest"]["run_id"],
                "successful_requests": len(records),
                "expected_requests": EXPECTED_REQUESTS_PER_TRIAL,
                "request_count_complete": len(records) == EXPECTED_REQUESTS_PER_TRIAL,
                "start_window_sample_count": len(start_samples),
                "starting_temperature_mean_c": _mean(start_temperatures),
                "starting_temperature_max_c": _maximum(start_temperatures),
                "starting_gpu_utilization_mean_pct": _mean(start_utilization),
                "starting_gpu_utilization_max_pct": _maximum(start_utilization),
                "start_gate_observed_from_artifacts": gate_observed,
                "median_output_tokens_per_second": _median(throughputs),
                "mean_output_tokens_per_second": _mean(throughputs),
                "median_engine_tpot_ms": _median(tpots),
                "first_20pct_output_tokens_per_second": first_throughput,
                "last_20pct_output_tokens_per_second": last_throughput,
                "first_to_last_throughput_change_pct": _percent_change(
                    first_throughput, last_throughput
                ),
                "first_20pct_engine_tpot_ms": first_tpot,
                "last_20pct_engine_tpot_ms": last_tpot,
                "first_to_last_tpot_change_pct": _percent_change(
                    first_tpot, last_tpot
                ),
                "mean_request_temperature_c": _mean(
                    _numbers(records, "gpu_temperature_mean_c")
                ),
                "max_request_temperature_c": _maximum(
                    _numbers(records, "gpu_temperature_max_c")
                ),
                "mean_sm_clock_mhz": _mean(
                    _numbers(records, "gpu_sm_clock_mean_mhz")
                ),
                "minimum_sm_clock_mhz": _minimum(
                    _numbers(records, "gpu_sm_clock_min_mhz")
                ),
                "thermal_limited_requests": sum(
                    row.get("gpu_limited_sw_thermal_seen") is True
                    or row.get("gpu_limited_hw_thermal_seen") is True
                    for row in records
                ),
                "thermal_limited_measurement_samples": sum(
                    sample.get("limited_sw_thermal") is True
                    or sample.get("limited_hw_thermal") is True
                    for sample in measurement_samples
                ),
                "measurement_wall_seconds": timing.get("measurement_wall_seconds"),
                "active_request_seconds": timing.get("active_request_seconds"),
            }
        )

    starting_means = _numbers(summaries, "starting_temperature_mean_c")
    coolest = min(starting_means) if starting_means else None
    for summary in summaries:
        start = summary["starting_temperature_mean_c"]
        summary["starting_temperature_within_3c"] = (
            isinstance(start, (int, float))
            and coolest is not None
            and float(start) - coolest <= MATCHED_START_TOLERANCE_C
        )
    return summaries


def _overall_summary(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starts = _numbers(trials, "starting_temperature_mean_c")
    trial_medians = _numbers(trials, "median_output_tokens_per_second")
    throughput_changes = _numbers(
        trials, "first_to_last_throughput_change_pct"
    )
    tpot_changes = _numbers(trials, "first_to_last_tpot_change_pct")
    return [
        {
            "completed_trials": len(trials),
            "complete_100_request_trials": sum(
                row["request_count_complete"] is True for row in trials
            ),
            "start_gate_verified_trials": sum(
                row["start_gate_observed_from_artifacts"] is True for row in trials
            ),
            "matched_start_temperature_trials": sum(
                row["starting_temperature_within_3c"] is True for row in trials
            ),
            "starting_temperature_min_c": _minimum(starts),
            "starting_temperature_max_c": _maximum(starts),
            "starting_temperature_range_c": (
                _maximum(starts) - _minimum(starts) if starts else None
            ),
            "median_of_trial_median_output_tokens_per_second": _median(
                trial_medians
            ),
            "mean_first_to_last_throughput_change_pct": _mean(
                throughput_changes
            ),
            "mean_first_to_last_tpot_change_pct": _mean(tpot_changes),
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze exp-002 continuous thermal drift runs."
    )
    parser.add_argument("--execution-id")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required; install the analysis extra with "
            "`python -m pip install -e .[analysis]`"
        ) from exc

    dataset = load_experiment_dataset(
        EXPERIMENT_DIR, execution_id=args.execution_id
    )
    if len(dataset.runs) != 3:
        raise ExperimentError(
            f"Execution contains {len(dataset.runs)} completed trials; expected 3"
        )
    measurements = _prepare_measurements(dataset.measurements)
    trials = _trial_summaries(dataset.runs, measurements)
    overall = _overall_summary(trials)
    results_dir = EXPERIMENT_DIR / "results"
    figures_dir = EXPERIMENT_DIR / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results_dir / "measurements.csv", measurements)
    write_csv(results_dir / "trial-summary.csv", trials)
    write_csv(results_dir / "aggregate.csv", overall)

    figures: list[str] = []
    runs = sorted(dataset.runs, key=_run_trial)
    figure, axes = plt.subplots(4, len(runs), figsize=(15, 11), sharex="col")
    if len(runs) == 1:
        axes = [[axis] for axis in axes]
    for column, run in enumerate(runs):
        trial = _run_trial(run)
        records = [
            row for row in measurements if int(row["trial_number"]) == trial
        ]
        request_x = [
            (float(row["started_offset_ms"]) + float(row["completed_offset_ms"]))
            / 2000.0
            for row in records
        ]
        telemetry = [
            sample
            for sample in run["telemetry"]
            if isinstance(sample.get("sample_offset_ms"), (int, float))
        ]
        telemetry_x = [float(sample["sample_offset_ms"]) / 1000.0 for sample in telemetry]
        axes[0][column].plot(
            request_x,
            [row.get("ollama_output_tokens_per_second") for row in records],
            color="#d95f02",
            linewidth=1.0,
            marker=".",
        )
        axes[0][column].set_title(f"Trial {trial}")
        axes[0][column].set_ylabel("Output tok/s")
        axes[1][column].plot(
            telemetry_x,
            [sample.get("temperature_c") for sample in telemetry],
            color="#e41a1c",
        )
        axes[1][column].set_ylabel("GPU temp (deg C)")
        axes[2][column].plot(
            telemetry_x,
            [sample.get("sm_clock_mhz") for sample in telemetry],
            color="#377eb8",
        )
        axes[2][column].set_ylabel("SM clock (MHz)")
        axes[3][column].plot(
            telemetry_x,
            [sample.get("gpu_utilization_pct") for sample in telemetry],
            color="#984ea3",
            label="Utilization %",
        )
        axes[3][column].plot(
            telemetry_x,
            [sample.get("power_draw_w") for sample in telemetry],
            color="#4daf4a",
            label="Power W",
        )
        axes[3][column].set_ylabel("Utilization / power")
        axes[3][column].set_xlabel("Run elapsed time (s)")
        axes[3][column].legend(fontsize=8)
        for row_index in range(4):
            axes[row_index][column].grid(alpha=0.2)
    figure.suptitle("Continuous inference: performance and GPU state")
    figures.extend(_save(figure, figures_dir, "continuous-thermal-timeline"))
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for trial in sorted({int(row["trial_number"]) for row in measurements}):
        records = [
            row
            for row in measurements
            if int(row["trial_number"]) == trial
            and row.get("status") == "success"
            and isinstance(row.get("normalized_throughput"), (int, float))
            and isinstance(row.get("ollama_engine_tpot_ms"), (int, float))
        ]
        request_numbers = [int(row["request_number"]) for row in records]
        normalized = [float(row["normalized_throughput"]) for row in records]
        tpots = [float(row["ollama_engine_tpot_ms"]) for row in records]
        axes[0].plot(
            request_numbers,
            _rolling_mean(normalized),
            label=f"Trial {trial}",
        )
        axes[1].plot(
            request_numbers,
            _rolling_mean(tpots),
            label=f"Trial {trial}",
        )
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Measured request number")
    axes[0].set_ylabel("Throughput / first-five mean")
    axes[1].set_xlabel("Measured request number")
    axes[1].set_ylabel("Engine TPOT (ms/token)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Five-request rolling performance trajectory")
    figures.extend(_save(figure, figures_dir, "normalized-performance-trajectory"))
    plt.close(figure)

    state_points = [
        row
        for row in measurements
        if all(
            isinstance(row.get(field), (int, float))
            for field in (
                "gpu_temperature_mean_c",
                "gpu_sm_clock_mean_mhz",
                "ollama_output_tokens_per_second",
                "request_number",
            )
        )
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = [float(row["request_number"]) for row in state_points]
    temp_plot = axes[0].scatter(
        [float(row["gpu_temperature_mean_c"]) for row in state_points],
        [float(row["ollama_output_tokens_per_second"]) for row in state_points],
        c=colors,
        cmap="viridis",
        alpha=0.8,
    )
    clock_plot = axes[1].scatter(
        [float(row["gpu_sm_clock_mean_mhz"]) for row in state_points],
        [float(row["ollama_output_tokens_per_second"]) for row in state_points],
        c=colors,
        cmap="viridis",
        alpha=0.8,
    )
    axes[0].set_xlabel("Mean request GPU temperature (deg C)")
    axes[0].set_ylabel("Output tokens/s")
    axes[1].set_xlabel("Mean request SM clock (MHz)")
    axes[1].set_ylabel("Output tokens/s")
    figure.colorbar(temp_plot, ax=axes[0], label="Request number")
    figure.colorbar(clock_plot, ax=axes[1], label="Request number")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("Throughput associations with GPU state")
    figures.extend(_save(figure, figures_dir, "throughput-vs-gpu-state"))
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    trial_labels = [f"Trial {int(row['trial_number'])}" for row in trials]
    positions = list(range(len(trials)))
    width = 0.36
    axes[0].bar(
        [position - width / 2 for position in positions],
        [
            _plot_number(row["first_20pct_output_tokens_per_second"])
            for row in trials
        ],
        width,
        label="First 20%",
        color="#1b9e77",
    )
    axes[0].bar(
        [position + width / 2 for position in positions],
        [
            _plot_number(row["last_20pct_output_tokens_per_second"])
            for row in trials
        ],
        width,
        label="Last 20%",
        color="#d95f02",
    )
    axes[1].bar(
        [position - width / 2 for position in positions],
        [_plot_number(row["first_20pct_engine_tpot_ms"]) for row in trials],
        width,
        label="First 20%",
        color="#1b9e77",
    )
    axes[1].bar(
        [position + width / 2 for position in positions],
        [_plot_number(row["last_20pct_engine_tpot_ms"]) for row in trials],
        width,
        label="Last 20%",
        color="#d95f02",
    )
    axes[0].set_ylabel("Output tokens/s")
    axes[1].set_ylabel("Engine TPOT (ms/token)")
    for axis in axes:
        axis.set_xticks(positions, trial_labels)
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    figure.suptitle("Early versus late continuous-inference performance")
    figures.extend(_save(figure, figures_dir, "first-vs-last-segments"))
    plt.close(figure)

    write_analysis_manifest(
        results_dir / "analysis-manifest.json", dataset, figures
    )
    print(f"Analyzed execution: {dataset.execution_id}")
    for row in trials:
        change = row["first_to_last_throughput_change_pct"]
        start = row["starting_temperature_mean_c"]
        print(
            f"trial {row['trial_number']}: "
            f"median={_format_number(row['median_output_tokens_per_second'])} "
            f"tok/s, first-to-last={_format_number(change)}%, "
            f"start={_format_number(start, 1)} deg C"
        )
    print(f"Results: {results_dir}")
    print(f"Figures: {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
