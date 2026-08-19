from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from inference_lab.analysis import (
    aggregate_conditions,
    load_experiment_dataset,
    write_analysis_manifest,
    write_csv,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent


def _numeric(records: list[dict[str, Any]], field: str) -> tuple[list[float], list[float]]:
    points = [
        (float(record["request_number"]), float(record[field]))
        for record in records
        if isinstance(record.get(field), (int, float))
        and not isinstance(record.get(field), bool)
    ]
    return [point[0] for point in points], [point[1] for point in points]


def _save(figure: Any, figures_dir: Path, stem: str) -> list[str]:
    paths: list[str] = []
    for extension in ("svg", "png"):
        path = figures_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path.relative_to(EXPERIMENT_DIR).as_posix())
    return paths


def _cooldown_spans(samples: list[dict[str, Any]]) -> list[tuple[float, float]]:
    offsets = [
        float(sample["sample_offset_ms"]) / 1000
        for sample in samples
        if sample.get("phase") == "cooldown"
        and isinstance(sample.get("sample_offset_ms"), (int, float))
    ]
    if not offsets:
        return []
    spans: list[tuple[float, float]] = []
    start = previous = offsets[0]
    for offset in offsets[1:]:
        if offset - previous > 1.5:
            spans.append((start, previous))
            start = offset
        previous = offset
    spans.append((start, previous))
    return spans


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze exp-001 thermal-soak runs.")
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
    measurements = list(dataset.measurements)
    aggregates = aggregate_conditions(dataset)
    results_dir = EXPERIMENT_DIR / "results"
    figures_dir = EXPERIMENT_DIR / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results_dir / "measurements.csv", measurements)
    write_csv(results_dir / "aggregate.csv", aggregates)

    colors = {"continuous": "#d95f02", "cooldown-60s": "#1b9e77"}
    figures: list[str] = []

    figure, axes = plt.subplots(4, len(dataset.runs), figsize=(14, 11), sharex="col")
    if len(dataset.runs) == 1:
        axes = [[axis] for axis in axes]
    for column, run in enumerate(dataset.runs):
        context = run["manifest"]["experiment"]
        condition_id = context["condition_id"]
        label = context["condition_label"]
        telemetry = run["telemetry"]
        run_measurements = [
            record for record in measurements if record["condition_id"] == condition_id
        ]
        request_x = [
            (float(record["started_offset_ms"]) + float(record["completed_offset_ms"]))
            / 2000
            for record in run_measurements
        ]
        throughput = [record["ollama_output_tokens_per_second"] for record in run_measurements]
        telemetry_x = [float(sample["sample_offset_ms"]) / 1000 for sample in telemetry]

        axes[0][column].plot(
            request_x,
            throughput,
            marker="o",
            color=colors.get(condition_id, "#333333"),
        )
        axes[0][column].set_title(label)
        axes[0][column].set_ylabel("Output tok/s")
        axes[1][column].plot(
            telemetry_x,
            [sample.get("temperature_c") for sample in telemetry],
            color="#e41a1c",
        )
        axes[1][column].set_ylabel("GPU temp (°C)")
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
        for row in range(4):
            axes[row][column].grid(alpha=0.2)
            for start, end in _cooldown_spans(telemetry):
                axes[row][column].axvspan(start, end, color="#bdbdbd", alpha=0.15)
    figure.suptitle("Sustained decode timeline (gray bands are cooldown periods)")
    figures.extend(_save(figure, figures_dir, "thermal-timeline"))
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for condition in dataset.spec.conditions:
        records = [
            record
            for record in measurements
            if record["condition_id"] == condition.condition_id
        ]
        axes[0].scatter(
            [record["gpu_temperature_mean_c"] for record in records],
            [record["ollama_output_tokens_per_second"] for record in records],
            label=condition.label,
            color=colors.get(condition.condition_id),
        )
        axes[1].scatter(
            [record["gpu_sm_clock_mean_mhz"] for record in records],
            [record["ollama_output_tokens_per_second"] for record in records],
            label=condition.label,
            color=colors.get(condition.condition_id),
        )
    axes[0].set_xlabel("Mean GPU temperature during request (°C)")
    axes[0].set_ylabel("Ollama output tokens/s")
    axes[1].set_xlabel("Mean SM clock during request (MHz)")
    axes[1].set_ylabel("Ollama output tokens/s")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Throughput associations with measured GPU state")
    figures.extend(_save(figure, figures_dir, "throughput-vs-gpu-state"))
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for condition in dataset.spec.conditions:
        records = [
            record
            for record in measurements
            if record["condition_id"] == condition.condition_id
        ]
        x, y = _numeric(records, "ollama_output_tokens_per_second")
        axes[0].plot(
            x,
            y,
            marker="o",
            label=condition.label,
            color=colors.get(condition.condition_id),
        )
    axes[0].set_xlabel("Measured request number")
    axes[0].set_ylabel("Ollama output tokens/s")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    x_positions: list[float] = []
    values: list[float] = []
    tick_labels: list[str] = []
    bar_colors: list[str] = []
    for index, aggregate in enumerate(aggregates):
        for offset, segment in ((-0.18, "first"), (0.18, "last")):
            x_positions.append(index + offset)
            values.append(
                float(aggregate[f"{segment}_segment_output_tokens_per_second"])
            )
            tick_labels.append(segment.title())
            bar_colors.append(colors.get(aggregate["condition_id"], "#777777"))
    axes[1].bar(x_positions, values, width=0.32, color=bar_colors)
    axes[1].set_xticks(
        [index for index in range(len(aggregates))],
        [aggregate["condition_label"] for aggregate in aggregates],
    )
    axes[1].set_ylabel("Mean output tokens/s")
    axes[1].set_title("First vs. last request quartile")
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle("Continuous and cooldown condition comparison")
    figures.extend(_save(figure, figures_dir, "condition-comparison"))
    plt.close(figure)

    write_analysis_manifest(
        results_dir / "analysis-manifest.json", dataset, figures
    )
    print(f"Analyzed execution: {dataset.execution_id}")
    for aggregate in aggregates:
        print(
            f"{aggregate['condition_id']}: "
            f"median={aggregate['median_output_tokens_per_second']:.2f} tok/s, "
            f"first-to-last={aggregate['first_to_last_change_pct']:.2f}%"
        )
    print(f"Results: {results_dir}")
    print(f"Figures: {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
