# Repository Guidance

Before changing project direction, experiment design, measurement semantics, or
workload formats, read the relevant material in `agent-docs/`.

- Keep `README.md` concise and useful to a public GitHub visitor.
- Keep detailed planning and persistent AI-agent context in `agent-docs/`.
- Keep each experiment's protocol, analysis, figures, results, and report together
  under `experiments/`.
- Keep raw prompts, generated outputs, telemetry captures, and run artifacts under
  ignored `runs/` unless a sanitized aggregate is deliberately published.
- Preserve the progression: benchmark, characterize, profile, hypothesize, modify,
  and remeasure.

Run deterministic checks from the repository root:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m ruff check .
```
