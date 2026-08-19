# Continuous Decode Performance as Laptop GPU Temperature Rises

**Status:** Planned  
**Experiment ID:** `exp-002-continuous-thermal-drift`  
**Execution ID:** Not run

## Question

During sustained, identical sequential decode requests, does active decode throughput
fall as GPU temperature rises, and is any change accompanied by lower SM clocks or
thermal/power clock-limiting events?

## Hypothesis

Continuous decode will raise GPU temperature. If thermal clock management constrains
the workload, later requests will have lower output tokens per second and higher
engine TPOT alongside reduced SM clocks and thermal-limiter events.

Temperature rising while clocks and decode performance remain stable would challenge
that mechanism. Performance drift without corresponding GPU-state changes would
suggest another cause, such as background load, power limiting, or runtime behavior.

## System Under Test

Complete from the execution manifest:

- Operating system and Python version: TBD
- Ollama version: TBD
- NVIDIA GPU, driver, and GPU UUID: TBD
- Model artifact, digest, parameter count, and quantization: TBD
- Git commit and working-tree state: TBD
- Laptop power, cooling, placement, and ambient conditions: TBD

## Method

### Workload

The study uses the public `decode-soak-v1` workload: one short, fixed synthetic prompt
with deterministic generation settings and a 256-token output limit. Reaching the
length limit is expected; this is a systems-load experiment, not a quality benchmark.
Generated text is not retained.

Each trial consists of:

- one excluded exact-prompt warmup;
- a post-warmup GPU-state start gate;
- 100 measured requests;
- concurrency 1;
- no delay between measured requests;
- required NVIDIA telemetry sampled every 500 ms;
- 30 seconds of telemetry before warmup and 60 seconds after measurement.

Three trials run in one execution. There is no fixed between-run sleep because each
run's start gate performs the state-based wait.

### Controlled variables

Keep the following fixed for the entire execution:

- exact model digest and quantization;
- Ollama and NVIDIA driver versions;
- prompt, prompt length, output limit, temperature, and seed;
- context-window and model offload configuration;
- single-request concurrency and model residency;
- AC power, Windows power mode, and laptop fan/performance mode;
- laptop surface, placement, and airflow;
- avoidable background CPU/GPU applications;
- telemetry interval.

Record ambient temperature if practical. Do not move the laptop or change its power
or cooling mode during the execution.

### Starting-state comparability

After the exact-prompt warmup, measurement begins only after ten consecutive 500 ms
samples report both:

- temperature at or below 70 degrees C;
- GPU utilization at or below 25%.

The gate times out after 900 seconds and fails the run instead of collecting a trial
from a noncomparable state. Analysis independently checks the ten telemetry samples
immediately preceding measured request one and reports their temperature and
utilization. It also reports whether the three mean starting temperatures fall within
3 degrees C of one another.

The threshold provides a common upper bound rather than guaranteeing identical
temperatures. A cross-trial starting-temperature spread above 3 degrees C will be
reported as a limitation and the experiment should be repeated before making a
cross-trial claim.

### Measurements

The primary performance metrics are Ollama engine output tokens per second and engine
TPOT, calculated as `eval_duration / eval_count`. Client E2E latency, TTFT, and client
TPOT are secondary measurements.

For every measured request, telemetry is interval-joined to derive temperature, SM
clock, utilization, power, and observed thermal/power limiter flags. Analysis compares
the first and last 20% of requests within each trial and preserves each trial as the
independent replicate. Requests are repeated observations within a trial, not 300
independent trials.

Every primary-analysis request must complete successfully with 256 output tokens and
`done_reason=length`. The analysis refuses an incomplete trial rather than silently
comparing unequal amounts of decode work; the private raw run still preserves the
failed or short request for diagnosis.

Evidence for a thermal mechanism requires more than a temperature correlation. The
expected mechanism chain is rising temperature, followed by thermal limiting or lower
SM clocks, followed by lower throughput and higher TPOT.

## Results

Not run. After execution, include:

- start-gate and cross-trial starting-temperature checks;
- request success and exact-length completion counts;
- per-trial median throughput and TPOT;
- first-to-last 20% changes;
- maximum temperature and minimum SM clock;
- thermal/power limiter observations;
- the generated timeline and association figures.

Expected analysis artifacts:

- `results/measurements.csv`
- `results/trial-summary.csv`
- `results/aggregate.csv`
- `results/analysis-manifest.json`
- `figures/continuous-thermal-timeline.{svg,png}`
- `figures/normalized-performance-trajectory.{svg,png}`
- `figures/throughput-vs-gpu-state.{svg,png}`
- `figures/first-vs-last-segments.{svg,png}`

## Interpretation

TBD. Report null or contradictory findings directly. Do not infer thermal causality
from temperature versus throughput alone.

## Limitations

- Temperature and elapsed time rise together, so the study is observational rather
  than a randomized temperature intervention.
- Three trials support replication but do not justify strong population-level claims.
- The start gate applies upper bounds; it does not force identical temperatures.
- `nvidia-smi` reports device-level samples at approximately 500 ms resolution, not
  individual kernels or decode steps.
- Desktop composition and other unavoidable laptop workloads may use the GPU.
- The fixed synthetic prompt measures reproducible decode load, not application
  quality or diverse production traffic.

Experiment 001's cooldown intervention is complementary evidence and should be used
when interpreting this continuous-only study.

## Conclusion

TBD after execution.

## Reproduction

Run from the repository root in PowerShell:

```powershell
.venv\Scripts\inference-lab.exe experiment validate `
  experiments\exp-002-continuous-thermal-drift

.venv\Scripts\inference-lab.exe experiment run `
  experiments\exp-002-continuous-thermal-drift

.venv\Scripts\python.exe `
  experiments\exp-002-continuous-thermal-drift\analysis.py
```

To reproduce analysis from a selected retained execution:

```powershell
.venv\Scripts\python.exe `
  experiments\exp-002-continuous-thermal-drift\analysis.py `
  --execution-id <execution-id>
```
