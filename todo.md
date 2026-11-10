# Current handoff

Implementation is underway per the user's request. Use `docs/plan.md`, `docs/tasks.md`, and `docs/reproducibility.md` for current scope and pending steps. The earlier setup-only instruction has been superseded by the explicit implementation request.

Next steps requiring model/data downloads and preferably free accelerator access: install `.[data,model,app,dev]`, run `scripts/prepare_data.py`, inspect the split manifest/counts without committing patient notes, train LoRA and select by validation loss, run baseline/adapter validation, then run final test once after settings are frozen. Run model-backed decode/attention benchmarks, update the report with real results, compile the PDF, record the demo, and submit externally.
