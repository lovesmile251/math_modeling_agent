# Answer Correctness Gold Gate

P0 optimization adds an explicit answer-correctness gate. Keep real gold files hidden during normal development; use `benchmarks/answer_correctness_gold.sample.json` only as the public schema example.

Supported checks:

- `expected_numeric_ranges`: key numeric answers must appear in result tables or paper text and fall inside the expected interval.
- `expected_decisions`: key choices, rankings, classifications, routes, or plans must appear in the paper or registered result tables.

Run reproduction audit with hidden gold:

```powershell
python -m tools.answer_reproduction_audit `
  --simulation-report benchmarks\results\contest_simulation.json `
  --output-dir benchmarks\results `
  --gold-expectations path\to\hidden_answer_gold.json
```

Run contest simulation with hidden gold:

```powershell
python -m tools.contest_simulation `
  --case cumcm-2024-c `
  --gold-expectations path\to\hidden_answer_gold.json
```

When gold is provided, answer correctness contributes to contest readiness and failed expectations block `first_prize_ready`. When gold is absent, the dimension is not applicable and score-neutral so legacy engineering drills still work.
