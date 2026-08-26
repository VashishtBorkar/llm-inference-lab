# Ollama Prefill Scaling With Natural-Language Prompt Length

**Status:** Preflight passed; ready for full run
**Experiment ID:** `exp-005-natural-corpus-prefill-scaling`
**Execution ID:** Not run

## Question

How do Ollama prompt-evaluation duration, prompt-processing throughput, client time
to first token, and total latency change as natural-language prompt length increases
on one RTX A6000?

## Hypothesis

Prompt-evaluation duration and client TTFT should increase as actual prompt-token
count increases. Prompt throughput may improve initially as larger prefills use the
GPU more fully, while attention work should eventually make duration grow more
rapidly. Nearly flat prompt-evaluation duration across the tested range would
challenge the hypothesis.

## System Under Test

The planned fixed configuration is:

- Ollama `0.32.15`
- `qwen2.5:7b-instruct`, expected local digest prefix `845dbda0ea48`
- Ollama's packaged Qwen2.5 7B Q4 model artifact
- One NVIDIA RTX A6000 with approximately 48 GB VRAM
- One request at a time and one loaded model
- Flash Attention enabled and f16 KV-cache storage
- A fixed 32,768-token context capacity for every request
- Prompt RAM caching and context checkpoints disabled at server launch

The execution manifests and private Ollama server log are authoritative for the
actual runtime, model digest, GPU UUID, driver, and observed telemetry.

## Method

The workload contains eight prompt sizes: 128, 256, 512, 1024, 2048, 4096,
8192, and 16,384 expected tokens. Source text comes from the raw WikiText-2 training
split, whose ignored local source file has SHA-256
`6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57`.
The generator removes heading-only and blank lines, normalizes WikiText punctuation
markers and whitespace, and verifies the source checksum before sampling.

Each prompt uses a deterministic contiguous window from a widely separated corpus
offset, preventing large shared prefixes between scenarios. The fixed instruction
`Reply with only OK.` appears after the sampled passage so it remains adjacent to
the generation boundary even for the 16K case. The response must exactly match
`OK`; an empty EOS-only response therefore remains an explicit quality failure.

Prompts are constructed and verified with the
`Qwen/Qwen2.5-7B-Instruct` tokenizer at revision
`a09a35458c702b33eeacc393d103063234e8bc28`. Analysis uses Ollama's reported
`prompt_eval_count` as the measured independent variable rather than assuming the
construction target is exact for the packaged model. Every scenario uses:

- context capacity 32,768;
- temperature 0 and seed 42;
- at most eight output tokens;
- thinking disabled;
- an exact-match `OK` response validator;
- concurrency one.

Short and long prompts alternate in workload order so prompt length does not grow
monotonically with request position. Each of three trials performs one complete
warmup pass followed by seven measured passes, producing 21 measured requests per
prompt length and 168 total measured requests. Warmups are excluded from analysis.

The model remains resident for steady-state timing. The launch procedure disables
the runner's prompt RAM cache and context checkpoints. Distinct corpus windows
prevent deliberate natural-language prefix reuse. This is a cache-state control,
not a cache-performance comparison.

Required `nvidia-smi` telemetry is sampled every 100 ms. GPU measurements are
supporting evidence for residency, utilization, clocks, power, and thermal state;
GPU memory is not a primary outcome because the context capacity remains fixed.

## Primary Measurements

- Ollama `prompt_eval_duration`, converted from nanoseconds to milliseconds
- Ollama `prompt_eval_count`
- Engine prompt throughput: `prompt_eval_count / prompt_eval_duration`

Supporting measurements are client TTFT, end-to-end latency, output count, failures,
GPU utilization, temperature, clocks, power, and process-level GPU memory.

The analysis aggregates individual measured requests by target prompt length. It
reports sample count, actual prompt-count range, median, mean, p95, observed range,
and standard deviation for prompt-evaluation duration, plus prompt-throughput and
client-latency summaries.

Planned outputs:

- `results/request-measurements.csv`
- `results/prompt-length-aggregate.csv`
- `results/analysis-manifest.json`
- `figures/prefill-scaling.svg` and `.png`
- `figures/client-latency-scaling.svg` and `.png`

## Results

The one-pass output preflight completed successfully on August 26, 2026. All eight
requests returned HTTP 200 and passed the exact-match `OK` validator, including the
16,384-token prompt. The preflight used no warmup, so its first 128-token request
included cold model loading and is not a scaling measurement. Private artifacts are
stored under ignored `runs/20260826T171513Z-exp005-preflight-8f92227b/`.

The full three-trial experiment has not run. Replace this section with the measured
prompt-duration curve, throughput curve, failures, and relevant GPU-state
observations after execution.

## Interpretation Guide

| Observation | Interpretation |
| --- | --- |
| Duration rises roughly proportionally while throughput is stable | Prefill cost scales approximately linearly over that range |
| Throughput rises at first | Larger prefills are using the GPU more efficiently |
| Duration bends upward and throughput falls at larger prompts | Attention or another length-dependent cost is growing faster than useful prefill work |
| Engine duration rises but TTFT rises substantially more | Client/runtime overhead outside measured prompt evaluation is also changing |
| Engine duration remains nearly flat | The hypothesis is challenged, or prompt reuse/cache control must be rechecked |

## Limitations

- Each prompt length uses one fixed corpus window. Length and passage content are
  therefore not independently randomized, and repetitions estimate timing variance
  rather than variation across natural-language samples.
- WikiText provides ordinary expository prose, but it is not semantically neutral;
  it inherits Wikipedia's topic selection, formatting artifacts, and biases.
- Regenerating the bundle requires the ignored, checksum-pinned WikiText source file.
- Results describe one quantized Ollama artifact, one runtime version, and one GPU;
  they are not a general model-family or engine comparison.
- The fixed 32K context allocation deliberately prevents this experiment from
  interpreting memory changes as per-token KV-cache growth.
- The shortest requests may complete between 100 ms telemetry samples. Ollama's
  engine timing remains available, but request-aligned GPU statistics may be absent.
- The runner environment controlling prompt caching is recorded in the protocol and
  private server log, not automatically captured as an Ollama API model property.
- Expected Hugging Face tokenizer counts may differ slightly from the packaged
  Ollama tokenizer/template. The analysis preserves and uses actual engine counts.

## Conclusion

To be completed after the experiment. State whether the measured prefill-duration
and throughput curves support or challenge the hypothesis.

## Reproduction

From a Slurm shell with one GPU, start the private Ollama server with the controlled
runtime configuration:

```bash
cd /common/home/vb471/llm-inference-lab
export PATH=/common/home/vb471/.local/bin:$PATH
export OLLAMA_MODELS=/common/home/vb471/model-cache/ollama
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NO_CLOUD=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_CONTEXT_LENGTH=32768
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=f16
export LLAMA_ARG_CACHE_RAM=0
export LLAMA_ARG_CTX_CHECKPOINTS=0

OLLAMA_LOG="runs/ollama-exp005-${SLURM_JOB_ID}.log"
ollama serve >"$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!
```

Validate the protocol, then run one measured pass as an output preflight:

```bash
.venv/bin/inference-lab experiment validate \
  experiments/exp-005-natural-corpus-prefill-scaling

.venv/bin/inference-lab run \
  --engine ollama \
  --base-url http://127.0.0.1:11434 \
  --model qwen2.5:7b-instruct \
  --workload workloads/synthetic/prefill-scaling-v2 \
  --warmup 0 \
  --repetitions 1 \
  --concurrency 1 \
  --timeout 600 \
  --gpu-telemetry-required
```

All eight preflight requests, especially `prompt-16384`, must report
`quality-ok`. Then run the complete experiment:

```bash
time .venv/bin/inference-lab experiment run \
  experiments/exp-005-natural-corpus-prefill-scaling
```

Analyze the latest complete execution:

```bash
.venv/bin/python \
  experiments/exp-005-natural-corpus-prefill-scaling/analysis.py
```

Stop Ollama after the run:

```bash
kill "$OLLAMA_PID"
wait "$OLLAMA_PID" 2>/dev/null || true
```
