from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from inference_lab import __version__
from inference_lab.engines.ollama import OllamaAdapter, OllamaError
from inference_lab.experiment import ExperimentError, load_experiment, run_experiment
from inference_lab.models import RequestRecord, RunConfig, TelemetryConfig
from inference_lab.runner import RunResult, StreamTimingError, run_benchmark
from inference_lab.telemetry import TelemetryError
from inference_lab.workload import WorkloadError, load_workload


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference-lab",
        description="Reproducible local LLM inference benchmarks.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-workload", help="Validate a workload bundle without running inference."
    )
    validate.add_argument("workload", type=Path)

    run = subparsers.add_parser("run", help="Run a benchmark workload.")
    run.add_argument("--engine", choices=["ollama"], default="ollama")
    run.add_argument("--base-url", default="http://127.0.0.1:11434")
    run.add_argument("--model", default="qwen3:4b-instruct")
    run.add_argument("--workload", type=Path, default=Path("workloads/smoke"))
    run.add_argument("--output-dir", type=Path, default=Path("runs"))
    run.add_argument("--warmup", type=_non_negative_integer, default=1)
    run.add_argument("--repetitions", type=_positive_integer, default=3)
    run.add_argument("--concurrency", type=_positive_integer, default=1)
    run.add_argument("--timeout", type=_positive_float, default=300.0)
    run.add_argument("--keep-alive", default="5m")
    run.add_argument("--label")
    run.add_argument("--inter-request-delay", type=_non_negative_float, default=0.0)
    run.add_argument("--gpu-telemetry", action="store_true")
    run.add_argument("--gpu-telemetry-required", action="store_true")
    run.add_argument("--gpu-sample-interval-ms", type=_positive_integer, default=500)
    run.add_argument("--gpu-pre-roll", type=_non_negative_float, default=0.0)
    run.add_argument("--gpu-post-roll", type=_non_negative_float, default=0.0)
    run.add_argument(
        "--capture-output",
        action="store_true",
        help="Include generated text in private run artifacts. Disabled by default.",
    )

    experiment = subparsers.add_parser(
        "experiment", help="Validate or run a specification-driven experiment."
    )
    experiment_commands = experiment.add_subparsers(
        dest="experiment_command", required=True
    )
    experiment_validate = experiment_commands.add_parser(
        "validate", help="Validate an experiment without running inference."
    )
    experiment_validate.add_argument("experiment", type=Path)
    experiment_run = experiment_commands.add_parser(
        "run", help="Run every condition and trial in an experiment."
    )
    experiment_run.add_argument("experiment", type=Path)
    return parser


def _format_number(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{decimals}f}"


def _progress(kind: str, record: RequestRecord) -> None:
    quality = "quality-ok" if record.quality_passed else "quality-failed"
    print(
        f"[{kind:8}] {record.scenario_id:24} "
        f"iteration={record.iteration:<3} status={record.status:<7} "
        f"ttft={_format_number(record.client_ttft_ms):>8} ms "
        f"e2e={_format_number(record.client_e2e_ms):>8} ms {quality}",
        flush=True,
    )


def _print_summary(result: RunResult) -> None:
    print("\nMeasured results (warmups excluded)")
    print(
        f"{'scenario':24} {'ok/total':>9} {'TTFT p50':>11} "
        f"{'E2E p50':>11} {'output tok/s':>13}"
    )
    for scenario_id, group in result.summary["by_scenario"].items():
        metrics = group["metrics"]
        print(
            f"{scenario_id:24} "
            f"{group['successful']}/{group['requests']:>7} "
            f"{_format_number(metrics['client_ttft_ms']['median']):>11} "
            f"{_format_number(metrics['client_e2e_ms']['median']):>11} "
            f"{_format_number(metrics['ollama_output_tokens_per_second']['median']):>13}"
        )
    overall = result.summary["overall"]
    print(
        f"\nRequest throughput: "
        f"{_format_number(overall['request_throughput_per_second'])} requests/s"
    )
    print(f"Artifacts: {result.run_dir}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-workload":
            workload = load_workload(args.workload)
            print(
                f"Valid workload: {workload.manifest['bundle_id']} "
                f"v{workload.manifest['bundle_version']}"
            )
            print(f"Scenarios: {len(workload.scenarios)}")
            print(f"SHA-256: {workload.content_sha256}")
            return 0

        if args.command == "experiment":
            spec = load_experiment(args.experiment, repo_root=Path.cwd())
            if args.experiment_command == "validate":
                print(f"Valid experiment: {spec.experiment_id}")
                print(f"Title: {spec.title}")
                print(f"Conditions: {len(spec.conditions)}")
                print(f"Trials per condition: {spec.trials_per_condition}")
                print(f"SHA-256: {spec.specification_sha256}")
                return 0
            print(
                f"Running {spec.experiment_id}: {len(spec.conditions)} conditions x "
                f"{spec.trials_per_condition} trial(s)",
                flush=True,
            )
            execution = run_experiment(spec, progress=_progress)
            print(f"Execution: {execution.execution_id}")
            print(f"Artifacts: {execution.execution_dir}")
            return 0

        config = RunConfig(
            model=args.model,
            workload_path=args.workload,
            output_root=args.output_dir,
            base_url=args.base_url,
            warmup=args.warmup,
            repetitions=args.repetitions,
            concurrency=args.concurrency,
            timeout_seconds=args.timeout,
            keep_alive=args.keep_alive,
            capture_output=args.capture_output,
            label=args.label,
            inter_request_delay_seconds=args.inter_request_delay,
            telemetry=TelemetryConfig(
                enabled=args.gpu_telemetry or args.gpu_telemetry_required,
                required=args.gpu_telemetry_required,
                interval_ms=args.gpu_sample_interval_ms,
                pre_roll_seconds=args.gpu_pre_roll,
                post_roll_seconds=args.gpu_post_roll,
            ),
        )
        adapter = OllamaAdapter(
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
        )
        result = run_benchmark(
            config=config,
            adapter=adapter,
            repo_root=Path.cwd(),
            progress=_progress,
        )
        _print_summary(result)
        failures = result.summary["overall"]["requests"] - result.summary["overall"][
            "successful"
        ]
        return 1 if failures else 0
    except (
        ExperimentError,
        WorkloadError,
        OllamaError,
        StreamTimingError,
        TelemetryError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
