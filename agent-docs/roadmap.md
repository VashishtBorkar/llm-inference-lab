# LLM Inference Lab — Systems Roadmap

## Project Direction

This lab uses repeatable, real-shaped workloads to study LLM inference as a system. Its purpose is not to produce a single engine leaderboard. It should build an evidence-backed understanding of how workload shape, model execution, serving policy, and hardware behavior affect latency, throughput, memory use, and output quality.

The project has three required outcomes:

1. controlled configuration experiments explained through model and serving behavior;
2. a profiler-led investigation of a reproducible bottleneck; and
3. one measured code change inside a serving-runtime execution path.

The lab defines and versions its own synthetic, public, and purpose-built representative workloads. Workload selection should serve a research question rather than couple the project to a separate application.

## Target Architecture

```text
Versioned workload bundle       Synthetic workload generator
            |                              |
            +--------------+---------------+
                           v
                    Benchmark runner
                           |
               Instrumentation + adapters
                           |
              +------------+------------+
              |            |            |
           Ollama         vLLM        SGLang
              |            |            |
              +------------+------------+
                           |
                       Model / GPU
                           |
                 Metrics + profiler data
```

The common harness should normalize requests and measurements without hiding useful engine-specific capabilities. Unsupported features and metrics must be recorded explicitly rather than fabricated or silently dropped.

Application admission control and engine scheduling are separate layers. The harness must support configurable in-flight concurrency so continuous batching and native scheduling remain observable.

## Experimental Rules

### Start with a hypothesis

Each study should state:

- the mechanism being tested;
- the primary variable being changed;
- the metrics expected to move;
- the variables held constant;
- the result that would disprove the hypothesis.

Prefer four to six meaningful studies over a large flag sweep.

### Keep clean benchmarks and profiling separate

Profiler instrumentation changes timing. Use clean runs to measure performance, separate profiler runs to explain it, and a final clean run to validate any change.

### Treat quality as a guardrail

An optimization is not successful if responses become malformed, ungrounded, truncated, or materially worse. Every relevant study must report correctness and quality alongside performance.

### Preserve provenance

Every result must identify the workload bundle, model, tokenizer, engine, configuration, software environment, hardware, cache state, warmup protocol, and measurement procedure that produced it. See [Benchmark Methodology](benchmark-methodology.md).

---

## Phase 1 — Measurement Contract and Ollama Baseline

**Status:** in progress. The core CLI, workload validation, Ollama streaming adapter, privacy-safe request records, environment manifest, aggregate summary, and deterministic tests are implemented. A larger repeated baseline and open-loop traffic generation remain before this phase is complete.

Build a UI-independent path that can run representative workloads against Ollama and persist one record per request.

Required capabilities:

- load and validate a versioned workload bundle;
- invoke an Ollama model with streaming enabled where supported;
- bound request concurrency and generate closed-loop or open-loop traffic;
- record timestamps, token counts, failures, and configuration provenance;
- keep raw prompt and output capture disabled by default;
- run task-specific correctness and structured-output validators;
- store raw request records separately from aggregate summaries.

Record at minimum:

- experiment, run, request, and scenario identifiers;
- engine and model identifiers or digests;
- task and workload class;
- prompt and generated token counts;
- request start, first token, and completion timestamps;
- time to first token and end-to-end latency;
- time per output token or inter-token timing when available;
- completion, cancellation, timeout, rejection, and failure state;
- engine-reported durations and token counts where available.

Representative baseline workload classes:

- long input with short structured extraction;
- long-document analysis with medium structured output;
- medium-input rewriting with short or medium output;
- long-context summarization or reasoning with longer output;
- short, latency-sensitive chat or editing;
- repeated shared prefixes;
- synthetic fixed-token prefill-heavy and decode-heavy cases.

**Completion criterion:** one command replays a versioned bundle against Ollama, emits privacy-safe per-request records and aggregate metrics, and produces stable repeated measurements for at least one workload from each major execution regime.

---

## Phase 2 — Workload Characterization

Establish baseline behavior before tuning the system.

Test at least three capability-appropriate points on each important axis:

```text
short / medium / long prompt
x
short / medium / long generation
x
single / moderate / saturated concurrency
```

Fixed maxima are not required. Out-of-memory conditions, rejection, queue growth, and saturation are useful boundaries when they are recorded accurately.

Produce curves for:

- prompt length versus time to first token;
- output length versus decode time and total latency;
- concurrency or arrival rate versus request and token throughput;
- concurrency versus queue time and p95 latency;
- prompt/output length versus memory use;
- throughput versus latency;
- shared-prefix ratio versus cache behavior and time to first token.

Use the results to classify workloads as primarily constrained by prefill, autoregressive decode, KV-cache capacity, memory bandwidth, CPU orchestration, queueing, batching, or another measured behavior.

**Completion criterion:** a characterization report identifies the dominant execution regime for each workload class and selects concrete questions for the controlled optimization studies.

---

## Phase 3 — Engine Adapters and Fair Comparisons

Add optimized serving engines behind the benchmark adapter boundary:

- Ollama remains the functional baseline;
- vLLM is a target optimized engine;
- SGLang is a target optimized engine.

Target all three when the selected model, operating system, and hardware are compatible. A recorded unsupported configuration is preferable to an invalid comparison.

Use two distinct comparison tracks.

### Controlled engine comparison

Hold constant, as far as the engines allow:

- physical hardware and power state;
- exact model and tokenizer revision;
- prompt bytes or chat template;
- weight and KV-cache precision;
- generation and sampling settings;
- input/output limits;
- workload order, warmup, and cache state.

This track is intended to isolate engine behavior. If model format or quantization differs, label the result as a stack comparison instead.

### Best practical stack comparison

Tune each engine independently within the same hardware and quality constraints. This track answers which complete configuration works best for a deployment goal, not which engine is intrinsically faster.

**Completion criterion:** the same versioned workload can run against each supported engine, and every comparison clearly states whether it is controlled or best-practical.

---

## Phase 4 — Configuration-Level Optimization

Run approximately four to six hypothesis-led studies. Start with one primary variable at a time, then test a combined tuned configuration only after the individual effects are understood.

### Study A — Prompt Length and Prefill

Vary prompt length and measure time to first token, prompt-processing throughput, total latency, utilization, and memory. Explain the curve through attention work, prefill execution, and KV-cache allocation.

### Study B — Output Length and Decode

Vary generated length and measure time per output token, inter-token latency, output throughput, total latency, and memory. Explain the result through autoregressive decode and memory movement.

### Study C — Concurrency, Batching, and Scheduling

Vary in-flight concurrency and arrival pattern. Measure throughput, queue time, time to first token, tail latency, rejection, memory, utilization, and fairness between short interactive and long requests.

Where supported, compare batching or scheduling policies. Explain the throughput versus tail-latency tradeoff rather than reporting throughput alone.

### Study D — Weight and KV-Cache Precision

Compare supported precision or quantization settings while keeping the model family and workload as constant as practical. Treat weight precision and KV-cache precision as separate variables.

Report latency, throughput, model and cache memory, maximum feasible context or concurrency, structured-output validity, grounding, and task quality.

### Study E — Prefix Caching and KV-Cache Management

Keep the concepts distinct:

- prefix caching reuses prefill work across requests that share tokens;
- KV-cache management governs allocation, paging, capacity, pressure, eviction, and preemption during inference.

Use repeated system prompts, document prefixes, or synthetic shared-prefix groups. Compare cold and warm states, caching enabled and disabled, and cache-aware policies where supported. Measure cache-hit tokens, time to first token, total latency, throughput, and memory.

### Study F — Engine Behavior

Use the controlled and best-practical tracks to compare Ollama, vLLM, and SGLang where feasible. Explain differences using observed scheduling, batching, caching, model format, kernels, or other evidence. Do not reduce the study to a winner table.

### Optional studies

Only when supported by the model, engine, and hardware:

- chunked prefill;
- speculative decoding;
- CUDA graphs versus eager execution;
- attention backend selection;
- CPU or KV-cache offload.

**Completion criterion:** four to six studies include the hypothesis, controlled variables, raw data, performance curves, mechanism-based explanation, quality checks, and known limitations.

---

## Phase 5 — Required Bottleneck Profiling

Select one reproducible performance issue from the earlier phases. Do not choose the runtime modification before this investigation.

Use a narrowing workflow:

```text
request metrics and engine telemetry
              -> system timeline
              -> framework/operator evidence
              -> kernel analysis only if warranted
```

Compatible tools may include engine metrics, OS/GPU telemetry, PyTorch Profiler, Nsight Systems, and Nsight Compute. Tool availability will depend on the engine and execution environment; the evidence requirement matters more than using every profiler.

Investigate the complete request path where relevant:

- client and transport overhead;
- queueing and engine scheduling;
- tokenization and CPU orchestration;
- CPU/GPU synchronization or GPU execution gaps;
- host/device memory movement;
- prefill and decode;
- batching, preemption, and request lifecycle;
- KV-cache allocation, eviction, and prefix-cache behavior;
- individual kernels only after higher-level evidence points to them.

The output of this phase must include:

- one precise bottleneck statement;
- a trace, profile, or correlated metric series supporting it;
- the implicated engine component or code path;
- a falsifiable hypothesis for a runtime change;
- a target metric and explicit regression guardrails.

**Completion criterion:** profiler evidence, not intuition, justifies one bounded modification inside a serving runtime.

---

## Phase 6 — One Runtime-Internal Modification

Make exactly one focused, evidence-selected behavioral change below the benchmark-client and launch-configuration layers.

Valid candidates include:

- an engine scheduling, priority, aging, or fairness policy;
- batching or chunked-prefill policy logic;
- prefix-cache or KV-cache eviction behavior;
- request lifecycle behavior inside the engine;
- a similarly bounded execution-path change tied to the profiled bottleneck.

The following do not satisfy the requirement by themselves:

- an API/provider adapter;
- a prompt or response-schema change;
- an engine launch flag;
- a benchmark-client-only queue or router;
- model selection or routing;
- instrumentation that does not change runtime behavior.

Keep the scope to one engine, one component, one hypothesis, and one patch. Upstream contribution is optional.

Validate with:

1. an unmodified upstream or pinned baseline;
2. the patched runtime under identical conditions;
3. repeated clean measurements without profiler overhead;
4. correctness and output-quality checks;
5. disclosed changes in latency, throughput, memory, fairness, reliability, and quality;
6. a causal explanation consistent with the profile and A/B result.

A null or negative result is acceptable if the experiment is sound and the explanation is honest.

**Completion criterion:** a reproducible A/B study shows the measured effect of the runtime change and explains why it did or did not solve the profiled bottleneck.

---

## Phase 7 — Synthesis and Adoption

Publish a coherent technical narrative:

```text
workload -> measured issue -> profile evidence -> hypothesis
         -> runtime change -> measured outcome -> tradeoffs
```

Required project outputs:

- benchmark harness and engine adapters;
- versioned workload specifications;
- experiment manifests and reproduction commands;
- privacy-safe raw measurements and aggregate results;
- performance curves and written analyses;
- the profiling investigation;
- the pinned runtime patch and controlled A/B study;
- correctness and quality results;
- a final report describing limitations and lessons learned.

The final report may recommend a safe default lab configuration. The experimental runtime patch may remain a pinned research fork; upstream contribution is optional.

---

## Optional Later Extension — Triton or CUDA

After the serving roadmap is complete, a separate extension may investigate one bounded transformer operation such as RMSNorm, a residual/normalization fusion, SiLU/SwiGLU-style fusion, softmax, or rotary embedding.

A possible progression is:

```text
PyTorch baseline -> Triton implementation -> profiler analysis
                 -> optional CUDA implementation -> end-to-end measurement
```

This is not a completion requirement and should not displace the profiler-driven serving-runtime modification.

## Scope Boundaries

### In scope

- local-first, single-node inference;
- a personal-device baseline and compatible single-GPU environments;
- one controlled model family or revision for engine comparisons;
- representative, public, and synthetic workloads;
- configuration studies tied to model and serving behavior;
- observability and profiling required by the experiments;
- one runtime-internal code modification.

### Out of scope

- application hosting or public multi-user deployment;
- Kubernetes, autoscaling, high availability, or multi-region systems;
- distributed or multi-node serving as a core requirement;
- training or fine-tuning;
- broad model leaderboards;
- production-fleet observability;
- building a full custom serving engine;
- multiple unrelated runtime modifications;
- Triton or CUDA kernels as a completion requirement.

Do not add infrastructure merely because large production systems use it. Prefer deeper inference understanding, controlled evidence, and reproducibility.
