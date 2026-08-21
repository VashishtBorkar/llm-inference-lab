# Benchmark Methodology

This document defines how experiments in the LLM Inference Lab should be designed, measured, and reported. The goal is reproducible explanation, not a single headline throughput number.

## 1. Benchmark Questions

Every experiment begins with a specific question and hypothesis. Examples:

- How does prompt length change prefill latency and memory use?
- At what concurrency does continuous batching improve throughput, and what happens to tail latency?
- Does a shared-prefix cache reduce time to first token for repeated profile context?
- Does quantization increase usable concurrency without unacceptable quality loss?
- Which measured bottleneck explains a difference between two serving configurations?

Record the expected mechanism, independent variable, controlled variables, target metrics, and falsifying result before running the study.

## 2. Workload Classes

The harness should support both representative and synthetic scenarios.

### Representative workloads

Use public datasets or purpose-built fixtures that cover common inference shapes:

| Class | Typical shape | Main behavior exercised |
| --- | --- | --- |
| Structured extraction | Long input, short structured output | Prefill and constrained output |
| Document analysis | Long input, medium structured output | Prefill, decode, schema validity |
| Rewriting | Medium input, short/medium output | Interactive latency |
| Summarization or reasoning | Long context, longer output | Mixed prefill and decode |
| Chat | Short input, short streamed output | TTFT and inter-token latency |
| Shared-prefix requests | Common prefix, varied suffix/output | Prefix-cache reuse |

### Synthetic isolation workloads

Synthetic cases should control token counts and isolate:

- short, medium, and long prefill;
- short and long decode;
- exact shared-prefix ratios;
- uniform and skewed prefix popularity;
- mixed short and long requests;
- fixed-rate arrivals and controlled bursts.

Synthetic results isolate mechanisms. Representative results show whether those mechanisms persist under more varied prompts and outputs.

## 3. Metric Definitions

Use consistent boundaries and record raw timestamps so derived metrics can be recomputed.

- **End-to-end latency:** client request start to final response receipt. Includes all client-visible waiting.
- **Server latency:** server acceptance to completion, when the engine exposes compatible timestamps.
- **Queue time:** time waiting for admission or engine scheduling. Identify the layer being measured.
- **Time to first token (TTFT):** client request start to receipt of the first generated token. If an engine-reported version is also available, store it separately.
- **Inter-token latency (ITL):** time between successive generated token arrivals. Preserve the distribution where streaming timestamps are available.
- **Time per output token (TPOT):** `(last_token_time - first_token_time) / (output_tokens - 1)` for responses with at least two generated tokens. Do not substitute total latency divided by token count.
- **Output-token throughput:** generated tokens per second over the stated measurement window.
- **Input-token throughput:** processed prompt tokens per second over the stated window.
- **Request throughput:** completed requests per second, with failed or rejected requests reported separately.
- **SLO goodput:** requests per second satisfying a declared latency and correctness objective.

Report at least median and p95 for request-level latency when sample size supports a tail estimate. Include sample count, mean or median, spread, failures, and the raw per-request data. Do not report p99 from too few samples.

Client-observed stream timing is not GPU-exact token execution time. Record how token
positions were identified, verify selected-token coverage against the engine's output
count, and report grouped or buffered stream events. When one event contains multiple
tokens, any within-event timing reconstruction is an estimate and must be labeled.

## 4. Resource and Quality Measurements

Performance results should include available resource measurements:

- peak and steady GPU memory;
- host memory;
- GPU utilization;
- power, temperature, and clocks when available;
- running and queued request counts;
- KV-cache utilization, hit tokens, eviction, or preemption where exposed.

Quality guardrails depend on the task and may include:

- JSON parse and schema-pass rate;
- required-field coverage;
- evidence grounding and unsupported-claim rate;
- retry or repair rate;
- truncation and incomplete-response rate;
- deterministic fixture checks or scored semantic evaluation.

Store quality outcomes per request so speed and correctness can be joined rather than reported from different samples.

## 5. Comparison Tracks

### Controlled comparison

Use this track to isolate an engine or configuration effect. Hold constant:

- hardware and operating environment;
- exact model and tokenizer revision;
- prompt bytes or chat template;
- weight and KV-cache precision;
- sampling and generation settings;
- input and output limits;
- workload bundle and request order;
- cold/warm state and warmup procedure.

If one of these differs, disclose it prominently and do not attribute the entire difference to the engine.

### Best-practical comparison

Tune each engine independently within the same hardware, model-quality, and reliability constraints. This compares deployable stacks. It does not isolate the serving engine.

Never merge controlled and best-practical results into one ranking.

## 6. Run Procedure

For each experiment:

1. validate the workload bundle and record its content hash;
2. capture the software, model, engine, and hardware manifest;
3. establish the declared cold or warm model and prefix-cache state;
4. run an explicit warmup that is excluded from reported samples;
5. run repeated measured trials;
6. preserve raw request records, including failures and rejected work;
7. run validators against the same outputs;
8. aggregate only after inspecting run-to-run stability;
9. repeat suspicious results before forming a conclusion.

For laptop experiments:

- use AC power and a consistent power mode;
- record temperature and clocks where possible;
- allow thermal settling between long runs;
- randomize or alternate configuration order;
- avoid unrelated foreground and background workloads;
- report run-to-run variance.

When comparable thermal starts are required, prefer a post-warmup state gate over a
fixed cooldown. Declare temperature and utilization thresholds, the number of
consecutive qualifying samples, and a timeout. The gate should use only samples
observed after it begins and fail before measurement if the state cannot be reached.

Thermal throttling and cold-start effects are part of the system, but they must not be mistaken for an engine change.

## 7. Traffic Models

Use a traffic model appropriate to the question:

- **single request:** isolates latency and model execution;
- **closed loop:** a fixed number of clients each sends the next request after completion;
- **open loop:** requests arrive independently at a declared rate, useful for queueing and overload behavior;
- **burst:** a bounded group arrives together, useful for batching and admission behavior.

Record arrival rate, concurrency cap, request-order seed, and burstiness. A global one-request-at-a-time client is unsuitable for studying continuous batching.

## 8. Cold, Warm, and Cache State

Label model state and prefix-cache state separately:

- cold model process versus resident/warmed model;
- cold prefix cache versus populated prefix cache;
- first-run compilation or graph capture versus steady state.

Do not average cold-start requests into steady-state latency without reporting them separately. For prefix-cache studies, define the shared token count or ratio and request popularity distribution.

## 9. Experiment Manifest

Each run should capture at least:

- experiment ID, run ID, timestamp, and code commit;
- workload format version, bundle ID/hash, and scenario IDs;
- engine name, version/commit, launch arguments, and adapter version;
- model and tokenizer artifact identifiers and revisions;
- chat template and prompt version;
- weight precision, quantization, and KV-cache dtype;
- sampling, stopping, seed, and length settings;
- concurrency, arrival pattern, request rate, and order seed;
- warmup, repetitions, duration, and cache state;
- OS, CPU, RAM, GPU, VRAM, driver, CUDA, and relevant libraries;
- power/thermal controls and known background load;
- supported and unavailable metrics;
- profiler status, which must be false for clean timing runs.

Pin versions rather than relying on mutable tags such as `latest`.

## 10. Profiling Protocol

Profiling follows a benchmark result; it does not replace one.

1. reproduce the issue with low-overhead metrics;
2. correlate request IDs with engine and system timelines;
3. localize the time to queueing, CPU orchestration, memory transfer, prefill, decode, cache behavior, or kernels;
4. use deeper tooling only at the layer supported by evidence;
5. write a falsifiable change hypothesis;
6. benchmark the baseline and modification again without profiler overhead.

Publish profiler captures only after checking them for prompts, generated text, usernames, local paths, and other sensitive data.

## 11. Privacy

- Committed/public workload fixtures must be synthetic.
- Sanitized or real-derived bundles require explicit opt-in and are private by default.
- Raw prompts, outputs, traces, and profiler captures are not publishable by default.
- Public per-request data uses opaque scenario IDs and contains metrics, not career content.
- Do not ingest private application databases or user data as benchmark input.

The canonical dataset rules are in [Workload Design](workload-design.md).

## 12. Reporting Results

Every report should include:

- the question and hypothesis;
- exact baseline and treatment configurations;
- workload and environment provenance;
- sample count, warmup, repetition, and traffic model;
- raw-data location and aggregation method;
- latency, throughput, resource, failure, and quality results;
- plots with units and clearly labeled axes;
- limitations and possible confounders;
- a mechanism-based interpretation;
- improvements and regressions.

Negative and null results are valid. Do not hide OOMs, rejected requests, malformed outputs, or configurations that failed to start.
