# Runtime Setup

The harness and inference server have separate lifecycles. Start the selected
server first, confirm its model identifier, and then run the benchmark client.

## Ollama

Install Ollama, pull the selected model, and leave the service running:

```powershell
ollama pull qwen3:4b-instruct
ollama serve
```

The default harness endpoint is `http://127.0.0.1:11434`.

## vLLM

vLLM is primarily supported on Linux and requires a compatible accelerator,
driver, PyTorch, and vLLM combination. Install it in a dedicated environment or
container rather than adding it to the lightweight benchmark-client environment.
Follow the current
[official vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/).

One basic launch pattern is:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 127.0.0.1 \
  --port 8000
```

The exact model must fit the available GPU memory. Choose a smaller compatible
instruction model or an appropriate supported precision/quantization when needed.
Record the exact model revision, dtype or quantization, vLLM version, launch command,
driver, and GPU in any experiment protocol.

Confirm the served identifier:

```bash
curl http://127.0.0.1:8000/v1/models
```

Then run the smoke workload:

```bash
inference-lab run \
  --engine vllm \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --workload workloads/smoke \
  --warmup 1 \
  --repetitions 3 \
  --concurrency 1
```

If the server requires a bearer token, pass `--api-key`. The key is used only in
the HTTP Authorization header and is not written to run artifacts.

The adapter uses the OpenAI-compatible `/v1/models` and streaming
`/v1/chat/completions` endpoints. vLLM usage counts populate prompt and output token
fields. TTFT, end-to-end latency, and stream timing remain client-observed. See the
[official server documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
for supported launch and request options.

### Experiment specifications

Set these defaults in an experiment only when the experiment is intended for vLLM:

```toml
[defaults]
engine = "vllm"
model = "Qwen/Qwen3-4B-Instruct-2507"

[defaults.engine_options]
base_url = "http://127.0.0.1:8000"
```

Do not put API keys in committed experiment files. The experiment runner currently
assumes a local unauthenticated endpoint; authenticated experiment execution should
be added through an environment-secret boundary if it becomes necessary.

## PyTorch reference

Install the optional dependencies into an environment with the appropriate PyTorch
build for the machine:

```bash
python -m pip install -e ".[pytorch]"
```

For CUDA systems, follow the official PyTorch installation selector when the
default package does not match the installed driver and CUDA environment.

Start with a small instruct model that fits comfortably in memory:

```bash
inference-lab run \
  --engine pytorch_reference \
  --model Qwen/Qwen3-0.6B \
  --device cuda \
  --dtype float16 \
  --workload workloads/smoke \
  --warmup 1 \
  --repetitions 1 \
  --concurrency 1
```

Use `--device cpu --dtype float32` to run without CUDA, though inference may be
slow. `--device auto` selects CUDA when available and otherwise selects CPU.

This backend intentionally accepts only `max_output_tokens` and greedy generation
(`temperature = 0`). It ignores `seed` because argmax has no randomness. Sampling,
stop strings, batching, scheduling, quantization, prefix caching, and server
lifecycle behavior are outside its learning scope. A workload requesting unsupported
generation behavior produces a failed request record instead of silently changing
semantics.

The model is loaded during run initialization and reused. Request timing therefore
describes steady-state inference rather than download or model-load time. Prefill
and decode durations are measured around direct model calls, with CUDA synchronization
at their timing boundaries. Concurrent requests are serialized; use concurrency one
when treating this backend as a simple reference.

An experiment specification may select it with:

```toml
[defaults]
engine = "pytorch_reference"
model = "Qwen/Qwen3-0.6B"

[defaults.engine_options]
device = "cuda"
dtype = "float16"
```

## Fair comparisons

Ollama model names often identify packaged or quantized artifacts, while vLLM model
names commonly identify Hugging Face repositories. Matching the model family name
is not enough for a controlled comparison. Verify the exact weights, tokenizer,
chat template, weight precision, context limit, generation settings, warmup, and
cache state. If these cannot be matched, describe the result as a complete-stack
comparison.
