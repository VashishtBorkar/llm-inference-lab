from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

plt.switch_backend("Agg")


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
BATCH_TOKENS = 1024
MAX_RETAINED_PREFIX_TOKENS = 32
ATTENTION_KERNELS_PER_CHUNK = 28
MATMUL_KERNELS_PER_CHUNK = 193
MLP_MATMUL_KERNELS_PER_CHUNK = 81
OTHER_QUANTIZED_MATMUL_KERNELS_PER_CHUNK = 112
MLP_ACTIVATION_KERNELS_PER_CHUNK = 27


class AnalysisError(RuntimeError):
    """Raised when a trace cannot be reconciled with its captured request."""


@dataclass(frozen=True)
class KernelEvent:
    start_ns: int
    end_ns: int
    name: str
    grid_x: int

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


def _kernel_events(database_path: Path) -> list[KernelEvent]:
    database = sqlite3.connect(database_path)
    try:
        rows = database.execute(
            """
            SELECT kernel.start, kernel.end, strings.value, kernel.gridX
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
            JOIN StringIds AS strings ON strings.id = kernel.demangledName
            ORDER BY kernel.start
            """
        ).fetchall()
    finally:
        database.close()
    return [
        KernelEvent(int(start), int(end), str(name), int(grid_x))
        for start, end, name, grid_x in rows
    ]


def _retained_prefix_tokens(log_path: Path) -> int:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    request_sections = re.split(r"\|\s+new prompt,", text)
    if len(request_sections) < 3:
        raise AnalysisError(f"{log_path} lacks warmup/profile cache accounting")
    measured_request = request_sections[-1]
    cached = re.search(r"cached n_tokens =\s*(\d+)", measured_request)
    if cached is None:
        raise AnalysisError(f"{log_path} lacks measured-request cache accounting")
    return int(cached.group(1))


def _family_events(
    events: list[KernelEvent],
    *,
    chunk_count: int,
) -> tuple[
    list[KernelEvent],
    list[KernelEvent],
    list[KernelEvent],
    list[KernelEvent],
]:
    matmul = [event for event in events if "void mul_mat_q<" in event.name]
    activation = [
        event for event in events if "unary_gated_op_kernel<&op_silu" in event.name
    ]
    if not matmul:
        raise AnalysisError("trace contains no batched quantized matmul kernels")

    # Decode uses the same Flash Attention kernel name but switches matmul to the
    # vector path. Keeping only Flash Attention launches before the final batched
    # matmul excludes the two-token response from the prefill comparison.
    last_prefill_matmul_end = max(event.end_ns for event in matmul)
    attention = [
        event
        for event in events
        if "flash_attn_ext_f16" in event.name
        and event.start_ns <= last_prefill_matmul_end
    ]

    # Each gated MLP has two expansion projections immediately before its SiLU
    # activation and one down projection immediately after it. Kernel launch
    # dimensions change with the short-prompt tile specialization, so temporal
    # order is more reliable here than a fixed grid shape or quantization type.
    mlp_matmul: list[KernelEvent] = []
    assigned: set[KernelEvent] = set()
    for silu in activation:
        expansion_projections = [
            event
            for event in matmul
            if event not in assigned and event.end_ns <= silu.start_ns
        ][-2:]
        if len(expansion_projections) != 2:
            raise AnalysisError("could not locate two MLP expansions before SiLU")
        mlp_matmul.extend(expansion_projections)
        assigned.update(expansion_projections)
        down_projection = next(
            (
                event
                for event in matmul
                if event not in assigned and event.start_ns >= silu.end_ns
            ),
            None,
        )
        if down_projection is None:
            raise AnalysisError("could not locate MLP down projection after SiLU")
        mlp_matmul.append(down_projection)
        assigned.add(down_projection)
    mlp_matmul.sort(key=lambda event: event.start_ns)
    other_quantized_matmul = [event for event in matmul if event not in assigned]

    expected = {
        "attention": chunk_count * ATTENTION_KERNELS_PER_CHUNK,
        "matmul": chunk_count * MATMUL_KERNELS_PER_CHUNK,
        "mlp_matmul": chunk_count * MLP_MATMUL_KERNELS_PER_CHUNK,
        "other_quantized_matmul": chunk_count * OTHER_QUANTIZED_MATMUL_KERNELS_PER_CHUNK,
        "activation": chunk_count * MLP_ACTIVATION_KERNELS_PER_CHUNK,
    }
    observed = {
        "attention": len(attention),
        "matmul": len(matmul),
        "mlp_matmul": len(mlp_matmul),
        "other_quantized_matmul": len(other_quantized_matmul),
        "activation": len(activation),
    }
    if observed != expected:
        raise AnalysisError(f"unexpected per-chunk kernel counts: {observed}; {expected=}")
    return attention, mlp_matmul, other_quantized_matmul, activation


def _duration_ms(events: list[KernelEvent]) -> float:
    return sum(event.duration_ns for event in events) / 1_000_000


def _chunk_durations(events: list[KernelEvent], chunk_count: int) -> list[float]:
    if len(events) % chunk_count:
        raise AnalysisError("kernel family cannot be divided evenly across chunks")
    kernels_per_chunk = len(events) // chunk_count
    return [
        _duration_ms(events[index * kernels_per_chunk : (index + 1) * kernels_per_chunk])
        for index in range(chunk_count)
    ]


def analyze_capture(metadata_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    prompt_tokens = int(metadata["prompt_tokens"])
    repetition = int(metadata["repetition"])
    profile = metadata["profile"]
    if profile["status"] != "success" or profile["prompt_eval_count"] != prompt_tokens:
        raise AnalysisError(f"invalid measured request in {metadata_path}")

    capture_dir = metadata_path.parent
    retained_prefix_tokens = _retained_prefix_tokens(capture_dir / "ollama.log")
    if not 0 < retained_prefix_tokens <= MAX_RETAINED_PREFIX_TOKENS:
        raise AnalysisError(
            f"unexpectedly retained {retained_prefix_tokens} tokens in {metadata_path}; "
            f"maximum allowed is {MAX_RETAINED_PREFIX_TOKENS}"
        )
    evaluated_tokens = prompt_tokens - retained_prefix_tokens
    chunk_count = math.ceil(evaluated_tokens / BATCH_TOKENS)

    events = _kernel_events(capture_dir / "nsight.sqlite")
    attention, mlp_matmul, other_quantized_matmul, activation = _family_events(
        events, chunk_count=chunk_count
    )
    matmul = mlp_matmul + other_quantized_matmul
    total_gpu_ms = _duration_ms(events)
    attention_ms = _duration_ms(attention)
    matmul_ms = _duration_ms(matmul)
    mlp_matmul_ms = _duration_ms(mlp_matmul)
    other_quantized_matmul_ms = _duration_ms(other_quantized_matmul)
    activation_ms = _duration_ms(activation)
    selected_ms = attention_ms + matmul_ms + activation_ms

    capture_row = {
        "prompt_tokens": prompt_tokens,
        "retained_prefix_tokens": retained_prefix_tokens,
        "evaluated_tokens": evaluated_tokens,
        "repetition": repetition,
        "chunk_count": chunk_count,
        "prompt_eval_ms": profile["prompt_eval_duration_ns"] / 1_000_000,
        "total_gpu_kernel_ms": total_gpu_ms,
        "attention_ms": attention_ms,
        "attention_instances": len(attention),
        "attention_avg_us": attention_ms * 1000 / len(attention),
        "attention_ms_per_evaluated_token": attention_ms / evaluated_tokens,
        "matmul_ms": matmul_ms,
        "matmul_instances": len(matmul),
        "matmul_avg_us": matmul_ms * 1000 / len(matmul),
        "matmul_ms_per_evaluated_token": matmul_ms / evaluated_tokens,
        "mlp_matmul_ms": mlp_matmul_ms,
        "mlp_matmul_instances": len(mlp_matmul),
        "mlp_matmul_avg_us": mlp_matmul_ms * 1000 / len(mlp_matmul),
        "mlp_matmul_ms_per_evaluated_token": mlp_matmul_ms / evaluated_tokens,
        "other_quantized_matmul_ms": other_quantized_matmul_ms,
        "other_quantized_matmul_instances": len(other_quantized_matmul),
        "other_quantized_matmul_avg_us": (
            other_quantized_matmul_ms * 1000 / len(other_quantized_matmul)
        ),
        "other_quantized_matmul_ms_per_evaluated_token": (
            other_quantized_matmul_ms / evaluated_tokens
        ),
        "mlp_activation_ms": activation_ms,
        "mlp_activation_ms_per_evaluated_token": activation_ms / evaluated_tokens,
        "attention_share_selected_pct": attention_ms / selected_ms * 100,
        "attention_share_total_gpu_pct": attention_ms / total_gpu_ms * 100,
    }

    attention_chunks = _chunk_durations(attention, chunk_count)
    mlp_matmul_chunks = _chunk_durations(mlp_matmul, chunk_count)
    other_quantized_matmul_chunks = _chunk_durations(
        other_quantized_matmul, chunk_count
    )
    activation_chunks = _chunk_durations(activation, chunk_count)
    chunk_rows: list[dict[str, Any]] = []
    for index in range(chunk_count):
        preceding_tokens = index * BATCH_TOKENS
        new_tokens = min(BATCH_TOKENS, evaluated_tokens - preceding_tokens)
        chunk_rows.append(
            {
                "prompt_tokens": prompt_tokens,
                "repetition": repetition,
                "chunk_index": index + 1,
                "preceding_evaluated_tokens": preceding_tokens,
                "new_tokens": new_tokens,
                "full_chunk": new_tokens == BATCH_TOKENS,
                "attention_ms": attention_chunks[index],
                "mlp_matmul_ms": mlp_matmul_chunks[index],
                "other_quantized_matmul_ms": other_quantized_matmul_chunks[index],
                "mlp_activation_ms": activation_chunks[index],
                "attention_ms_per_new_token": attention_chunks[index] / new_tokens,
                "mlp_matmul_ms_per_new_token": (
                    mlp_matmul_chunks[index] / new_tokens
                ),
            }
        )
    return capture_row, chunk_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AnalysisError(f"refusing to write empty result {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def aggregate_lengths(captures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in captures:
        by_length[int(row["prompt_tokens"])].append(row)

    metrics = (
        "prompt_eval_ms",
        "total_gpu_kernel_ms",
        "attention_ms",
        "attention_avg_us",
        "attention_ms_per_evaluated_token",
        "matmul_ms",
        "matmul_avg_us",
        "matmul_ms_per_evaluated_token",
        "mlp_matmul_ms",
        "mlp_matmul_avg_us",
        "mlp_matmul_ms_per_evaluated_token",
        "other_quantized_matmul_ms",
        "other_quantized_matmul_avg_us",
        "other_quantized_matmul_ms_per_evaluated_token",
        "mlp_activation_ms",
        "mlp_activation_ms_per_evaluated_token",
        "attention_share_selected_pct",
        "attention_share_total_gpu_pct",
    )
    aggregated: list[dict[str, Any]] = []
    for prompt_tokens, rows in sorted(by_length.items()):
        result: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "evaluated_tokens": rows[0]["evaluated_tokens"],
            "chunk_count": rows[0]["chunk_count"],
            "captures": len(rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_min"] = min(values)
            result[f"{metric}_max"] = max(values)
        aggregated.append(result)
    return aggregated


def _save_figure(figure: plt.Figure, stem: str) -> None:
    for suffix in ("png", "svg"):
        destination = FIGURES_DIR / f"{stem}.{suffix}"
        figure.savefig(destination, dpi=180, bbox_inches="tight")
        if suffix == "svg":
            svg = destination.read_text(encoding="utf-8")
            destination.write_text(
                "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(figure)


def plot_kernel_scaling(aggregated: list[dict[str, Any]]) -> None:
    x = [row["prompt_tokens"] for row in aggregated]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    axes[0].plot(
        x,
        [row["mlp_matmul_ms_mean"] for row in aggregated],
        "o-",
        label="MLP matmul",
    )
    axes[0].plot(
        x,
        [row["other_quantized_matmul_ms_mean"] for row in aggregated],
        "o-",
        label="Other quantized matmul",
    )
    axes[0].plot(
        x,
        [row["attention_ms_mean"] for row in aggregated],
        "o-",
        label="Flash Attention",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Actual prompt tokens")
    axes[0].set_ylabel("Total GPU kernel time (ms)")
    axes[0].set_title("Kernel-family scaling")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        x,
        [row["mlp_matmul_ms_per_evaluated_token_mean"] for row in aggregated],
        "o-",
        label="MLP matmul",
    )
    axes[1].plot(
        x,
        [
            row["other_quantized_matmul_ms_per_evaluated_token_mean"]
            for row in aggregated
        ],
        "o-",
        label="Other quantized matmul",
    )
    axes[1].plot(
        x,
        [row["attention_ms_per_evaluated_token_mean"] for row in aggregated],
        "o-",
        label="Flash Attention",
    )
    axes[1].plot(
        x,
        [row["mlp_activation_ms_per_evaluated_token_mean"] for row in aggregated],
        "o-",
        label="MLP SiLU activation",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Actual prompt tokens")
    axes[1].set_ylabel("GPU kernel time per evaluated token (ms)")
    axes[1].set_title("Normalized kernel cost")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(
        x,
        [row["attention_share_selected_pct_mean"] for row in aggregated],
        "o-",
        color="tab:red",
    )
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("Actual prompt tokens")
    axes[2].set_ylabel("Flash Attention share (%)")
    axes[2].set_title("Share of selected prefill kernels")
    axes[2].grid(alpha=0.25)

    figure.suptitle("Ollama prefill kernel scaling on RTX A6000")
    figure.tight_layout()
    _save_figure(figure, "kernel-scaling")


def plot_long_context_chunks(chunks: list[dict[str, Any]]) -> None:
    longest = max(int(row["prompt_tokens"]) for row in chunks)
    selected = [row for row in chunks if int(row["prompt_tokens"]) == longest]
    by_chunk: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_chunk[int(row["chunk_index"])].append(row)

    x = sorted(by_chunk)
    attention = [statistics.mean(float(row["attention_ms"]) for row in by_chunk[i]) for i in x]
    mlp_matmul = [
        statistics.mean(float(row["mlp_matmul_ms"]) for row in by_chunk[i])
        for i in x
    ]
    other_quantized_matmul = [
        statistics.mean(
            float(row["other_quantized_matmul_ms"]) for row in by_chunk[i]
        )
        for i in x
    ]
    activation = [
        statistics.mean(float(row["mlp_activation_ms"]) for row in by_chunk[i]) for i in x
    ]
    figure, axis = plt.subplots(figsize=(8.5, 5))
    axis.plot(x, mlp_matmul, "o-", label="MLP matmul")
    axis.plot(
        x,
        other_quantized_matmul,
        "o-",
        label="Other quantized matmul",
    )
    axis.plot(x, attention, "o-", label="Flash Attention")
    axis.plot(x, activation, "o-", label="MLP SiLU activation")
    axis.set_xlabel("1,024-token chunk index")
    axis.set_ylabel("GPU kernel time per chunk (ms)")
    axis.set_title(f"Kernel cost across one {longest:,}-token prefill")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, "long-context-chunk-scaling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Experiment 6 Nsight traces")
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metadata_paths = sorted(run_dir.glob("prompt-*/repeat-*/capture.json"))
    if not metadata_paths:
        raise AnalysisError(f"no profile captures found in {run_dir}")

    captures: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for metadata_path in metadata_paths:
        capture, capture_chunks = analyze_capture(metadata_path)
        captures.append(capture)
        chunks.extend(capture_chunks)
    captures.sort(key=lambda row: (int(row["prompt_tokens"]), int(row["repetition"])))
    chunks.sort(
        key=lambda row: (
            int(row["prompt_tokens"]),
            int(row["repetition"]),
            int(row["chunk_index"]),
        )
    )
    aggregated = aggregate_lengths(captures)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(RESULTS_DIR / "kernel-capture-summary.csv", captures)
    _write_csv(RESULTS_DIR / "kernel-length-aggregate.csv", aggregated)
    _write_csv(RESULTS_DIR / "chunk-measurements.csv", chunks)
    manifest = {
        "schema_version": "1.0",
        "experiment_id": "exp-006-ollama-prefill-kernel-scaling",
        "execution_id": run_dir.name,
        "capture_count": len(captures),
        "prompt_lengths": [row["prompt_tokens"] for row in aggregated],
        "observed_retained_prefix_tokens": sorted(
            {row["retained_prefix_tokens"] for row in captures}
        ),
        "maximum_allowed_retained_prefix_tokens": MAX_RETAINED_PREFIX_TOKENS,
        "batch_tokens": BATCH_TOKENS,
        "kernel_classification": {
            "attention": "demangled name contains flash_attn_ext_f16; decode launches excluded by final batched-matmul timestamp",
            "quantized_matmul": "demangled name contains 'void mul_mat_q<'",
            "mlp_matmul": "two preceding expansion projections plus the first batched matmul after each SiLU activation",
            "other_quantized_matmul": "remaining batched quantized matmuls; dominated by attention-side projections but not assigned exclusively from Nsight Systems evidence",
            "mlp_activation": "demangled name contains unary_gated_op_kernel<&op_silu",
        },
        "raw_profiler_artifacts_committed": False,
    }
    (RESULTS_DIR / "analysis-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    plot_kernel_scaling(aggregated)
    plot_long_context_chunks(chunks)
    print(f"Analyzed {len(captures)} captures from {run_dir.name}")
    print(f"Wrote sanitized results to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
