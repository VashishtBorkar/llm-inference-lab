# LLM Inference Lab

An experiment-driven project for understanding and improving LLM inference systems.

## Project Overview

This repository contains a reproducible benchmark harness and a growing collection
of controlled LLM inference experiments. The goal is to understand how workload
shape, hardware state, and serving-engine behavior affect latency, throughput,
memory, and output quality.

The harness currently provides:

- specification-driven, repeatable experiments;
- per-request TTFT, TPOT, throughput, token, and failure measurements;
- synchronized NVIDIA telemetry for temperature, clocks, utilization, power,
  memory, and limiter events; and
- versioned workloads, manifests, analysis scripts, figures, and reports.

Ollama is the current baseline engine. vLLM and SGLang are planned as optimized
serving backends. The long-term objective is to move from characterization into
profiling and make one evidence-driven change inside a serving runtime.


## Results

### Experiment 002: Burst Versus Sustained Inference

Experiment 002 found that short-run performance on a GTX 1660 Ti laptop was not
representative of sustained serving capacity.

| Metric | Initial burst | Sustained segment |
| --- | ---: | ---: |
| Output throughput | 66-68 tokens/s | 17-20 tokens/s |
| Mean SM clock | 1.46-1.50 GHz | 0.47-0.52 GHz |

Across three trials, software thermal limiting reduced clock speed and late-segment
throughput was 61-72% below the early segment. The result demonstrates why inference
capacity should be measured in its sustained operating regime rather than inferred
from a few fast initial requests.

[Read the complete experiment report](experiments/exp-002-continuous-thermal-drift/report.md).

![Burst-to-sustained inference transition](experiments/exp-002-continuous-thermal-drift/figures/continuous-thermal-timeline.png)

## Experiments

| Experiment | Question | Status |
| --- | --- | --- |
| [Thermal soak](experiments/exp-001-thermal-soak/report.md) | Does cooldown change active decode performance? | Preliminary study complete |
| [Continuous thermal drift](experiments/exp-002-continuous-thermal-drift/report.md) | How different are burst and sustained inference rates? | Complete |
| [Intra-request decode latency](experiments/exp-003-intra-request-decode-latency/report.md) | Do later tokens in one long response arrive more slowly? | Protocol ready |

Each experiment keeps its specification, analysis code, committed aggregate results,
figures, and report together under [`experiments/`](experiments/README.md). Private raw
runs are excluded from Git.

## Quick start

Requirements:

- Python 3.11 or newer
- Ollama running locally
- `qwen3:4b-instruct` installed in Ollama

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[analysis]"
ollama pull qwen3:4b-instruct

.venv\Scripts\inference-lab.exe validate-workload workloads\smoke
.venv\Scripts\inference-lab.exe run `
  --engine ollama `
  --model qwen3:4b-instruct `
  --workload workloads\smoke `
  --warmup 1 `
  --repetitions 3 `
  --concurrency 1
```

Run a versioned experiment with:

```powershell
.venv\Scripts\inference-lab.exe experiment validate `
  experiments\exp-002-continuous-thermal-drift

.venv\Scripts\inference-lab.exe experiment run `
  experiments\exp-002-continuous-thermal-drift
```

## Repository layout

```text
src/inference_lab/   benchmark runner, metrics, telemetry, and engine adapters
workloads/           versioned synthetic and representative workloads
experiments/         specifications, analysis code, results, and reports
agent-docs/          detailed roadmap and development context for AI agents
tests/               deterministic tests that do not require Ollama
runs/                private raw executions, excluded from Git
```

## Roadmap

The next stages are workload-scaling studies on university GPU compute, vLLM and
SGLang adapters, concurrency and caching experiments, profiler-led bottleneck
analysis, and one measured change inside a serving-runtime execution path.

Detailed planning and methodology live in [`agent-docs/`](agent-docs/README.md).

## Tests

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m ruff check .
```
