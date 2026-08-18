# LLM Inference Lab

An independent, experiment-driven project for learning how large language model inference systems behave, where their bottlenecks come from, and how serving performance can be improved.

The lab combines controlled synthetic workloads with public or purpose-built representative workloads. Its experiments are designed around inference behavior rather than any one application.

## Goals

The project has three required learning goals:

1. **Optimize inference through configuration and model understanding.** Run controlled experiments and explain the results using prefill, autoregressive decode, batching, scheduling, KV-cache behavior, memory use, and model execution.
2. **Profile the system to find bottlenecks.** Trace measured performance problems through the request path and use profiler evidence before selecting an optimization.
3. **Make one change below the client and configuration layers.** Modify one serving-runtime execution path, then validate the change with controlled A/B measurements and correctness checks.

The intended progression is:

```text
benchmark -> characterize -> profile -> hypothesize -> modify -> remeasure -> explain
```

## Ollama Baseline Harness

The first vertical slice is implemented as a reproducible, UI-independent benchmark harness for the current Ollama system. It can:

- replay versioned workloads;
- capture per-request latency, token, throughput, and failure data;
- replay prompt/output shapes with configurable repetitions, closed-loop concurrency, and model keep-alive;
- preserve complete experiment manifests and raw results;
- validate structured-output correctness and task quality alongside speed.

Ollama is the functional baseline. vLLM and SGLang are planned optimized-engine targets once the harness and measurement contract are stable.

### Quick start

Requirements:

- Python 3.11 or newer;
- a running local Ollama server;
- the benchmark model installed in Ollama.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
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

The smoke bundle contains short-control, prefill-heavy, and decode-heavy synthetic scenarios. It validates the measurement path; it is not a model-quality benchmark.

Each run creates a private, ignored artifact directory:

```text
runs/<run-id>/
  manifest.json       # engine, model, workload, environment, and run settings
  requests.jsonl      # one raw record per warmup and measured request
  summary.json        # aggregates computed from measured requests only
```

Generated text is hashed but not stored unless `--capture-output` is explicitly supplied. The harness records client-observed TTFT and end-to-end latency alongside Ollama's engine-reported token counts and duration fields. Stream events are counted but are not assumed to correspond one-to-one with tokens.

### Tests

The harness has no runtime package dependencies. Run the deterministic test suite without Ollama:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Experimental Focus

The repository will contain:

- independent workload definitions, replay, and synthetic load generation;
- Ollama, vLLM, and SGLang benchmark adapters and launch configurations;
- experiment manifests, measurement code, profiler tooling, results, and plots;
- configuration-level optimization studies;
- the profiler-motivated runtime modification.

## Scope

The core scope is local-first, single-node LLM inference with one controlled model family where practical. It includes representative and synthetic workloads, configuration experiments, serving-engine comparisons, profiling, and one focused runtime-internal modification.

The following are not completion requirements:

- training or fine-tuning;
- broad model leaderboards;
- Kubernetes, autoscaling, high availability, or multi-region deployment;
- distributed or multi-node serving;
- building a complete serving engine;
- multiple unrelated runtime forks;
- Triton or CUDA kernel work.

Kernel work may become a later extension after the serving roadmap is complete.

## Documentation

- [Systems Roadmap](docs/roadmap.md) — phases, experiments, completion criteria, and scope boundaries.
- [Benchmark Methodology](docs/benchmark-methodology.md) — workloads, metrics, controls, reproducibility, quality, and reporting rules.
- [Workload Design](docs/workload-design.md) — portable scenario format, workload families, validation, and dataset handling.

## Status

The Phase 1 measurement spine and Ollama vertical slice are implemented. The next milestone is to run a larger repeated Ollama baseline, inspect run-to-run stability, and expand the synthetic workload matrix before beginning vLLM or SGLang adapters.

## Guiding Principle

> Use controlled workloads to characterize, profile, and optimize LLM inference across serving engines and runtime configurations, then make one evidence-driven runtime change and explain why performance changed.
