# Assignment 2 TODO

## Done in this repository

- [x] Add leakage-safe TrialGPT data preprocessing and the deterministic 37/8/8 patient split generator.
- [x] Add LoRA fine-tuning and validation/test evaluation scripts.
- [x] Add classification and evidence metrics for all six expert labels.
- [x] Implement reference and tiled online-softmax attention.
- [x] Implement uncached, KV-cached, and greedy speculative decoding.
- [x] Add the Streamlit screening app and interactive inference benchmark page.
- [x] Draft the IEEE LaTeX report and reproducibility instructions.
- [x] Run local CPU correctness checks (8 tests pass).

## Next: prepare data and environment

- [ ] On an environment with dataset access, install project dependencies: `pip install -e '.[data,model,app,dev]'`.
- [ ] Run `python scripts/prepare_data.py`.
- [ ] Check the generated patient counts and leakage check in `data/splits/patient_split_seed42.json`.
- [ ] Keep patient notes and generated data out of Git; commit only the split manifest if appropriate.
- [ ] Confirm the free GPU has enough memory for the 1.54B target, 0.49B draft, and LoRA training.

## Train and evaluate

- [ ] Run `python scripts/train_lora.py` on the training split.
- [ ] Inspect training logs, validation loss, and examples affected by 4096-token truncation.
- [ ] Run baseline and adapter evaluation on validation: `python scripts/evaluate.py --split validation` (add `--adapter PATH` for the adapter).
- [ ] Review label and evidence errors; document at least eight distinct real failures, including missing-information and unsupported-evidence cases.
- [ ] Freeze model and settings before final test evaluation.
- [ ] Run the held-out test once: `python scripts/evaluate.py --split test --final-test`.
- [ ] Save and review real metrics and prediction artifacts; do not report unrun results.

## Validate inference methods

- [ ] Run `python scripts/benchmark_attention.py`; record numerical error, runtime, and memory on the available hardware.
- [ ] Run `python scripts/benchmark_decoding.py`; verify token equality for target greedy, cached, and speculative decoding.
- [ ] Record prefill/decode time, throughput, peak memory, draft acceptance, and any speculative slowdown.
- [ ] Check that target and draft tokenizer vocabularies match before speculative decoding.

## Finish app and submission materials

- [ ] Start the app with `streamlit run app.py` in an environment that permits local network sockets.
- [ ] Try an unseen patient note and inclusion/exclusion criteria; verify cited sentence text and JSON export.
- [ ] Run the inference comparison page with downloaded target and draft model weights.
- [ ] Update the IEEE report with actual methods, results, figures/tables, error analysis, author details, repository URL, and AI-use acknowledgment.
- [ ] Compile `report/main.tex` to PDF and review the PDF.
- [ ] Capture application screenshots and record the required 5–7 minute demo.
- [ ] Publish the repository and provide a model adapter download path.
- [ ] Submit the report/project through Google Classroom and attend the live evaluation.

## Current blockers

- This workspace has CPU-only PyTorch and no Transformers/PEFT installation; no dataset or model weights have been downloaded.
- The sandbox denied Streamlit's local socket bind, so app startup must be checked in a normal local or hosted environment.
- `pdflatex` is unavailable here, so the report PDF must be compiled elsewhere.
