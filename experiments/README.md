# Experiments

Each experiment is a versioned investigation with a declared question, controlled
conditions, reproducible analysis, and a written result. Raw executions live under
`runs/` and are intentionally excluded from Git. Sanitized aggregates, figures, and
reports are committed with the experiment.

## Workflow

```powershell
inference-lab experiment validate experiments\exp-001-thermal-soak
inference-lab experiment run experiments\exp-001-thermal-soak
python experiments\exp-001-thermal-soak\analysis.py
```

Copy `experiment-template.toml` and `report-template.md` when starting a study.

## Experiment Index

| ID | Question | Status | Report |
| --- | --- | --- | --- |
| `exp-001-thermal-soak` | Does cooldown change sustained decode performance on a laptop GPU? | Completed preliminary study: cooldown lowered maximum temperature from 92°C to 83°C, while active decode throughput remained similar | [Report](exp-001-thermal-soak/report.md) |
