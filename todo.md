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
- [x] Run `python scripts/prepare_data.py` in Colab.
- [x] Verify split: 37/8/8 patients, 603/206/206 annotations, leakage check passed. Manifest is currently in the Colab runtime; copy it into the repository later.
- [ ] Keep patient notes and generated data out of Git; commit only the split manifest if appropriate.
- [ ] Confirm the free GPU has enough memory for the 1.54B target, 0.49B draft, and LoRA training.

## Train and evaluate

- [x] Run `python scripts/train_lora.py` on the T4 GPU for 3 epochs (228 steps; about 11m37s).
- [x] Save/download the trained adapter zip from Colab. It is not yet in this repository.
- [ ] Preserve the training log and inspect examples affected by 4096-token truncation.
- [x] Run baseline and adapter evaluation on the same 206 validation examples. Baseline: accuracy 0.262, macro-F1 0.111, evidence micro-F1 0. Adapter: accuracy 0.723, macro-F1 0.356, evidence micro-F1 0.429. Adapter label outputs are strongly biased toward `not excluded`; investigate before drawing conclusions.
- [x] Review eight validation label/evidence errors. All eight were gold `not enough information` predicted as `not excluded`; several had irrelevant or incomplete evidence IDs. The sample spans only three patient notes, so it is not eight independent patient cases. Review more diverse cases before final conclusions.
- [ ] Freeze model and settings before final test evaluation.
- [ ] Run the held-out test once: `python scripts/evaluate.py --split test --final-test`.
- [ ] Save and review real metrics and prediction artifacts; do not report unrun results.

## Validate inference methods

- [x] Run `python scripts/benchmark_attention.py` on Colab Tesla T4. All six causal/noncausal cases passed numerical comparison (maximum absolute error 5.36e-7). Tiled implementation was slower in every tested case (2.58x–51.98x); measured memory differences were small and mixed. Results are recorded in `docs/reproducibility.md`.
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

## Latest Colab run (Nov. 10, 2026)

- GPU: Tesla T4; training used FP16 after a Colab-only script adjustment for Transformers 5 and T4 precision.
- Training completed: train loss 0.1347; final validation loss 0.04293. These are language-model losses, not eligibility metrics.
- The adapter zip was downloaded by the user. It still needs to be copied into the project artifacts / a durable model location.
- Baseline/adapter validation JSON was generated in Colab; keep it private/out of Git because it contains patient notes and predictions. The held-out test split remains untouched.
- User-reported validation values: inclusion n=66, accuracy 0.727, macro-F1 0.250; exclusion n=140, accuracy 0.721, macro-F1 0.140.
- The first Colab decoder benchmark attempt was interrupted (`KeyboardInterrupt`) during its initial uncached generation; no benchmark JSON or decoder metrics were produced. Target/draft weights are now cached in that runtime.
- The Colab runtime may expire when idle. The downloaded zip is safe; reconnect and restore the repo/data/model environment to continue.
