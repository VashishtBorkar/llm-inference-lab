# Engine Architecture

The benchmark runner communicates with inference systems through adapters in
`src/inference_lab/engines/`. Adapters translate engine-neutral workload scenarios
and return a common `GenerationObservation`; they must not manufacture unsupported
server metrics.

## Supported adapters

- `OllamaAdapter` calls Ollama's native streaming chat API. It exposes Ollama's
  server-side prefill and decode durations and token counts.
- `VllmAdapter` calls a separately managed vLLM OpenAI-compatible server. It
  records client-observed stream timing and usage token counts. The OpenAI API does
  not expose the same per-request prefill and decode duration fields as Ollama, so
  those engine-duration metrics remain unavailable.
- `PyTorchReferenceAdapter` runs one Transformers causal language model in the
  benchmark process. It contains an explicit full-prompt prefill followed by a
  one-token-at-a-time greedy decode loop that passes the KV cache between calls.

The adapter factory is the only place the CLI and specification-driven experiment
runner select a concrete adapter. Each adapter declares capabilities that are
written into the run manifest. A missing capability or metric is represented as
unavailable, never inferred from an incompatible measurement.

## Configuration boundary

Workloads remain engine-neutral. `RunConfig` contains common benchmark settings and
one `engine_options` mapping. The factory validates that mapping against the selected
engine: HTTP endpoints belong to server adapters, `keep_alive` belongs to Ollama,
and device/dtype belong to the PyTorch reference. Server launch arguments belong
outside workload files and must be recorded with the experiment protocol when they
are experimental variables.

Experiment schema 1.1 stores these values under `[defaults.engine_options]` (and,
when needed, `[conditions.run.engine_options]`). This replaces the engine-specific
`base_url` and `keep_alive` keys that previously appeared in the common defaults
table.

The harness does not start or stop serving processes. This prevents startup,
model-loading, and server-management behavior from becoming an implicit part of a
clean benchmark window.

## Metric compatibility

Request record version 1.1 adds engine-neutral duration and throughput names. The
older `ollama_*` fields remain as compatibility aliases so existing committed
experiment analyses continue to work. They are empty for non-Ollama adapters.

Client TTFT, end-to-end latency, TPOT, request throughput, token counts, failures,
quality validation, and GPU telemetry use the common measurement path. Server-side
durations are included only when the serving API provides them.

## PyTorch reference

The PyTorch backend is intentionally a teaching implementation, not a custom
serving engine. It is kept in one readable adapter with this explicit sequence:

```text
render chat -> tokenize -> move tensors -> prefill -> choose token
            -> one-token decode loop using the KV cache -> decode text
```

Transformers loads the tokenizer, model implementation, and weights. The adapter
owns the visible generation loop, uses batch size one, and supports only greedy
generation. It does not add queues, dynamic batching, replicas, or a server layer.
Concurrent harness requests are serialized around the single model instance. Those
omitted mechanisms are precisely the hidden complexity this reference is meant to
contrast with Ollama and vLLM.

Read `src/inference_lab/engines/pytorch_reference.py` from `_load` through
`generate`. The numbered comments in `generate` mark prompt rendering, tokenization,
prefill, and cached decode. It calls the model directly and deliberately does not
use `model.generate`.
