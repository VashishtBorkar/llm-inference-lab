# Experiments

Each experiment is a versioned investigation with a declared question, controlled
conditions, reproducible analysis, and a written result. Raw executions live under
`runs/` and are intentionally excluded from Git. Sanitized aggregates, figures, and
reports are committed with the experiment.

## Workflow

```powershell
.venv\Scripts\inference-lab.exe experiment validate `
  experiments\exp-002-continuous-thermal-drift
.venv\Scripts\inference-lab.exe experiment run `
  experiments\exp-002-continuous-thermal-drift
.venv\Scripts\python.exe `
  experiments\exp-002-continuous-thermal-drift\analysis.py
```

Copy `experiment-template.toml` and `report-template.md` when starting a study.
`experiment run` validates automatically. Analysis stays experiment-local because
each study has different aggregation and figure requirements.

## Experiment Index

| ID | Question | Status | Report |
| --- | --- | --- | --- |
| `exp-001-thermal-soak` | Does cooldown change sustained decode performance on a laptop GPU? | Completed preliminary study: cooldown lowered maximum temperature from 92 C to 83 C, while active decode throughput remained similar | [Report](exp-001-thermal-soak/report.md) |
| `exp-002-continuous-thermal-drift` | Does continuous decode slow as GPU temperature rises, and do clocks or limiter events explain the change? | Ready to run: 3 trials of 100 sequential 256-token requests | [Protocol](exp-002-continuous-thermal-drift/report.md) |
| `exp-003-intra-request-decode-latency` | Does client-observed token interarrival latency change across one 4,096-token streamed response? | Ready to run: 5 thermally matched long-response trials | [Protocol](exp-003-intra-request-decode-latency/report.md) |
