# Output-Token Latency Across One Long Streamed Response

**Status:** Planned  
**Experiment ID:** `exp-003-intra-request-decode-latency`  
**Execution ID:** Not run

## Question

As one autoregressive response grows from its first generated token to token 4096,
does client-observed output-token interarrival latency increase? If it does, is the
change associated with sequence position, rising GPU temperature, falling SM clocks,
or NVIDIA clock-limiting events?

## Hypothesis

Later tokens may arrive more slowly because each autoregressive decode step operates
over a longer sequence and larger KV cache. Increasing inter-token latency while GPU
clocks remain stable would support this sequence-length explanation. A latency rise
that coincides with higher temperature, thermal limiting, and lower SM clocks would
instead support a thermal mechanism. A flat latency curve would challenge both
slowdown explanations over the tested decode range.

## System Under Test

To be completed from the execution manifests:

- Operating system, Python version, Ollama version, and NVIDIA driver
- GPU model, UUID, VRAM, and telemetry field support
- Exact `qwen3:4b-instruct` model digest and quantization
- Ollama context window of 8192 tokens
- AC-power, Windows power-mode, cooling-mode, airflow, and ambient-temperature notes

## Method

The experiment contains five independent trials. Each trial has:

1. One 64-token warmup using the same synthetic prompt. This loads and warms the
   model while verifying that privacy-safe selected-token timing covers Ollama's
   reported `eval_count`.
2. A post-warmup start gate requiring ten consecutive 500 ms GPU samples at or below
   70 C and 25% utilization. The gate times out after 1200 seconds rather than
   silently starting a thermally unmatched trial.
3. One measured request targeting exactly 4096 generated tokens with temperature 0,
   seed 42, thinking disabled, concurrency 1, and no request-level cooldown.
4. Sixty seconds of post-roll telemetry.

Each measured request must complete successfully with `eval_count=4096`,
`done_reason=length`, and complete selected-token stream coverage. The analysis
refuses incomplete trials rather than comparing unequal response lengths.

Raw token identities, generated text, and log-probability values are not retained.
The harness stores only stream-event timestamps, selected-token counts, character
counts, and request identifiers. When one stream event contains multiple selected
tokens, its client arrival interval is divided evenly among those tokens and the
affected observations are flagged as grouped estimates.

## Aggregation

The first generated token is excluded from inter-token latency because its timestamp
contains prompt evaluation and time to first token. Tokens 2 through 4096 are divided
into 256-token-position windows. The analysis calculates a mean client inter-token
latency for each trial and window, then aggregates those five trial-level values.
Individual token intervals are not treated as independent experimental replicates.

Primary outputs:

- `results/window-measurements.csv`: one row per trial and token window
- `results/window-aggregate.csv`: across-trial summaries by token window
- `results/trial-summary.csv`: request and first-versus-last-quarter summaries
- `results/analysis-manifest.json`: execution provenance, aggregation unit, and warnings

Planned figures:

- Client inter-token latency versus generated-token position
- Latency, GPU temperature, and SM clock aligned by token position
- Cumulative client TPOT at successive response checkpoints
- Latency associations with temperature and SM clock
- First-quarter versus last-quarter latency for every trial

## Results

Not run. Replace this section with numerical findings, failures, coverage checks, and
the strongest generated figures after executing and analyzing the experiment.

## Interpretation Guide

| Observation | Interpretation supported by the measurement |
| --- | --- |
| Latency rises while temperature and clocks remain stable | Sequence-length or KV-cache growth is a stronger explanation |
| Latency rises with thermal flags and falling clocks | Thermal clock management is a stronger explanation |
| Client events become grouped or bursty while Ollama's request-average decode rate stays stable | Stream delivery, buffering, or client scheduling may contribute |
| Latency remains flat across token position | Long response time is mainly due to generating more tokens over this range |
| Latency, temperature, and sequence position all rise together | Both mechanisms remain plausible; this experiment alone cannot assign causality |

Experiment 002 independently characterizes sustained thermal drift across repeated
fixed-length requests and should be used as complementary evidence.

## Limitations

- Client-observed inter-token latency includes Ollama serialization, local transport,
  Python iteration, and operating-system scheduling; it is not a direct GPU kernel
  timestamp.
- Requesting selected-token logprobs may add processing or serialization overhead.
  Every trial uses the same setting, so the within-experiment curve remains
  controlled, but absolute streaming pace may differ from an uninstrumented request.
- Token position and elapsed time are inseparable within one response. GPU temperature
  may therefore rise with sequence length even when both affect latency.
- NVIDIA telemetry is sampled every 500 ms, much more slowly than individual decode
  steps. GPU statistics are joined to 256-token windows, not individual tokens.
- Equal spacing inside a multi-token stream event is an estimate. The analysis reports
  the grouped-token fraction and does not represent those estimates as exact timings.
- Five trials support a repeatability assessment but not high-confidence tail-latency
  claims or a general result across models, engines, or hardware.
- The synthetic response is deliberately length-terminated and does not evaluate
  semantic quality.

## Conclusion

To be completed after the experiment. State whether the evidence supports, challenges,
or fails to distinguish the sequence-length and thermal explanations.

## Reproduction

Validate the specification without contacting Ollama:

```powershell
.venv\Scripts\inference-lab.exe experiment validate `
  experiments\exp-003-intra-request-decode-latency
```

Run the experiment manually when the laptop is on AC power, avoidable GPU workloads
are closed, and the cooling configuration has been recorded:

```powershell
.venv\Scripts\inference-lab.exe experiment run `
  experiments\exp-003-intra-request-decode-latency
```

Analyze the latest complete matching execution:

```powershell
.venv\Scripts\python.exe `
  experiments\exp-003-intra-request-decode-latency\analysis.py
```

Pass `--execution-id <id>` to reproduce analysis for a particular retained execution.
