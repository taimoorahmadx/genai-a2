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
- [x] Keep patient notes and generated data out of Git; generated artifacts are ignored.
- [x] Confirm the free T4 ran LoRA training. The adapter archive is available locally as an ignored file; do not commit it.

## Train and evaluate

- [x] Run `python scripts/train_lora.py` on the T4 GPU for 3 epochs (228 steps; about 11m37s).
- [x] Save/download the trained adapter zip from Colab. A local ignored copy exists; do not add it to Git.
- [x] Check training-log availability: only final train/validation losses were reported; step-wise logs were not retained and cannot be reconstructed. Truncation effects remain a limitation to disclose.
- [x] Run baseline and adapter evaluation on the same 206 validation examples. Baseline: accuracy 0.262, macro-F1 0.111, evidence micro-F1 0. Adapter: accuracy 0.723, macro-F1 0.356, evidence micro-F1 0.429. Adapter label outputs are strongly biased toward `not excluded`; investigate before drawing conclusions.
- [x] Review eight validation label/evidence errors. All eight were gold `not enough information` predicted as `not excluded`; several had irrelevant or incomplete evidence IDs. The sample spans only three patient notes, so it is not eight independent patient cases. Review more diverse cases before final conclusions.
- [x] Freeze the current model/settings for final evaluation; do not tune from test results.
- [x] Run the held-out test exactly once after freezing settings using the saved adapter on Kaggle. Overall: n=206, accuracy 0.6893, macro-F1 0.3439, evidence micro-F1 0.4419; inclusion n=75 accuracy/macro-F1 0.680/0.235; exclusion n=131 accuracy/macro-F1 0.695/0.137. The model predicted `not excluded` for every exclusion case. No gold `excluded` cases were present in test.
- [x] Review aggregate metrics. Keep `test_finetuned.json` private; it contains patient notes and row-level predictions. Do not tune or rerun on test.

## Validate inference methods

- [x] Run `python scripts/benchmark_attention.py` on Colab Tesla T4. All six causal/noncausal cases passed numerical comparison (maximum absolute error 5.36e-7). Tiled implementation was slower in every tested case (2.58x–51.98x); measured memory differences were small and mixed. Results are recorded in `docs/reproducibility.md`.
- [x] Run a reduced Colab T4 decoding smoke benchmark at input/output lengths 32/4, 64/4, and 128/8. All three methods produced identical target greedy token IDs at every length. One-repeat timings are preliminary; full measurements are in `docs/reproducibility.md`.
- [x] Record throughput, peak allocated memory, draft acceptance, and relative timing for the reduced run. KV cache was fastest; speculative decoding was slower than KV cache and accepted all proposed tokens in these short cases.
- [x] Verify target/draft tokenizer compatibility in Colab: vocabularies match and special token IDs match. The benchmark script now checks both before speculative decoding.

## Finish app and submission materials

- [ ] Start the app with `streamlit run app.py` in an environment that permits local network sockets.
- [ ] Try an unseen patient note and inclusion/exclusion criteria; verify cited sentence text and JSON export.
- [ ] Run the inference comparison page with downloaded target and draft model weights.
- [x] Fill report metadata (Taimoor Ahmad, Computer Science, FAST NUCES, repository URL); measured validation/benchmark results, error analysis and AI-use acknowledgement are included.
- [x] Add system workflow diagram and document that step-wise training logs were not retained, so a training curve cannot be recreated honestly.
- [ ] Compile `report/main.tex` to PDF and review it; no LaTeX compiler is installed in this environment.
- [ ] Capture application screenshots and record the required 5–7 minute demo.
- [ ] Publish the repository and provide a model adapter download path.
- [ ] Submit the report/project through Google Classroom and attend the live evaluation.

## Current blockers

- This workspace has CPU-only PyTorch and no Transformers/PEFT installation; the trained adapter zip is local but ignored, while base/draft model weights and the dataset are unavailable locally.
- The sandbox denied Streamlit's local socket bind, so app startup must be checked in a normal local or hosted environment.
- `pdflatex` is unavailable here, so the report PDF must be compiled elsewhere.

## Latest Colab run (Nov. 10, 2026)

- GPU: Tesla T4; training used FP16 after a Colab-only script adjustment for Transformers 5 and T4 precision.
- Training completed: train loss 0.1347; final validation loss 0.04293. These are language-model losses, not eligibility metrics.
- The adapter zip was downloaded and is available locally as an ignored file. It is intentionally not in Git; preserve it somewhere durable for re-upload to Colab.
- Baseline/adapter validation and final test JSON were generated in Colab/Kaggle; keep them private/out of Git because they contain patient notes and predictions. The test was evaluated once with settings frozen.
- User-reported validation values: inclusion n=66, accuracy 0.727, macro-F1 0.250; exclusion n=140, accuracy 0.721, macro-F1 0.140.
- An initial Colab decoder benchmark attempt was interrupted during uncached generation. A reduced smoke run later completed; its one-repeat measurements are preliminary, and the explicit tokenizer equality check has since passed. Target/draft weights are cached in that runtime.
- The Colab runtime may expire when idle. The downloaded zip is safe; reconnect and restore the repo/data/model environment to continue.
- Kaggle GPU access was enabled for the final test run; no further GPU work is essential for the core evaluation result.
- The final test was successfully evaluated once on Kaggle after removing incompatible `torchao` 0.10.0. Aggregate output is documented; keep row-level JSON private and do not rerun/tune on test.
- Time-critical next action: compile/review the report PDF if a LaTeX environment is available, then record the demo and submit through Classroom.
