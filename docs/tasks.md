# Assignment checklist

Status symbols: **Implemented** means repository code/docs exist; **Validated locally** means checked with small deterministic CPU examples, not real model experiments; **Pending run** requires downloading data/model and/or accelerator execution; **External** requires a human action outside the repository.

## Research and setup

- [x] Inspect environment and hardware: Linux x86_64, Python 3.10.12, PyTorch 2.14.1 CPU-only; no CUDA. Transformers/PEFT absent. Setup and limits recorded in README/reproducibility doc.
- [x] Collect five credible sources on the assignment: TrialGPT study, dataset card, Qwen/Phi model cards, LoRA, FlashAttention and speculative decoding. References are in report draft; verify all citations and add discussion when preparing final paper.
- [x] Compare decoder candidates: Qwen2.5-1.5B-Instruct target (Apache-2.0) and Qwen2.5-0.5B-Instruct draft (Apache-2.0; same family tokenizer checked at runtime); Phi-3.5-mini-instruct (3.8B, MIT) alternative. Model facts and memory lower bounds documented.
- [ ] Download model card revisions/license text and verify free accelerator memory / compatible tokenizer files on actual execution environment.

## Data, training and evaluation

- [x] Schema allowlist preprocessing: prior GPT-4 fields cannot flow into prompts or train rows; six labels and expert sentence IDs retained.
- [x] Deterministic patient split/manifest implementation and no-leak checks (37/8/8, NumPy seed 42).
- [x] Run dataset preparation in Colab: 37/8/8 patients, 603/206/206 annotations; leakage check passed. Copy the split manifest into the repo later; keep patient notes out of git.
- [x] Implement LoRA supervised training with answer-only loss, structured output, documented prompt/truncation/config.
- [x] GPU LoRA training completed on Colab Tesla T4 for 3 epochs / 228 steps; reported train loss 0.1347 and best/final validation loss 0.04293. The adapter zip is present locally but ignored by Git; training log is not archived and truncation still needs review.
- [x] Implement baseline/adapter evaluation on identical selected split, class/evidence metrics, inclusion/exclusion breakdowns and empty evidence policy.
- [x] Run baseline and adapter on the same 206 validation annotations. Baseline accuracy/macro-F1 0.262/0.111; adapter 0.723/0.356. Adapter evidence micro-F1 0.429 (baseline 0). Adapter is biased toward `not excluded`; per-class failure review remains required.
- [x] One-time final test evaluation after settings were frozen, run on Kaggle. Fine-tuned adapter: n=206, accuracy 0.6893, macro-F1 0.3439, evidence micro-F1 0.4419; inclusion (n=75) accuracy/macro-F1 0.680/0.235; exclusion (n=131) 0.695/0.137. It predicted `not excluded` for every exclusion example; no gold `excluded` cases occurred in test. Do not tune or rerun on test.
- [x] Include per-example predictions and error fields in evaluation artifacts; review eight validation error examples. All eight gold `not enough information` labels were predicted as `not excluded`; several cited irrelevant or incomplete evidence. They came from only three patient notes, so these are eight criterion-level errors, not eight independent patient cases. Broader, patient-diverse error analysis remains advisable.

## Inference engineering

- [x] Implement uncached reference and explicit KV-cache greedy decoding; local mock verifies exact token equality at three output lengths.
- [ ] Verify against actual target model at three prompt/output lengths and benchmark synchronized repeated runs, prefill/decode latency, throughput and peak memory.
- [x] Implement reference scaled dot-product attention and tiled online-softmax attention.
- [x] CPU numerical tests cover three shapes and causal/noncausal masks; max/mean error computed by benchmark script.
- [x] Execute attention timing/memory experiment on Colab Tesla T4; all six cases passed equivalence checks (maximum absolute error 5.36e-7). Tiled attention was 2.58x–51.98x slower on these small shapes, with small/mixed peak-memory differences. Full measurements are in `docs/reproducibility.md`.
- [x] Implement greedy speculative decoding including target verification, correction after rejection, output/cache synchronization by re-prefill; local deterministic mock verifies mismatch correction.
- [x] Verify actual Qwen target, KV-cache, and speculative generated token IDs match at three reduced prompt/output lengths on Colab T4; record preliminary acceptance, speed, and memory results. Target/draft vocabulary and special token IDs also match. A longer repeated benchmark is still needed for stable performance estimates.

## Application, paper and handoff

- [x] Build standalone Streamlit screening workflow with evidence, missing information, explicit aggregation, educational notice and report export.
- [x] Build interactive model-backed inference comparison page; it measures uncached, cached and speculative paths and verifies output equality when run.
- [x] Add setup/reproduction instructions, optional dependencies, data handling and model download path.
- [x] Create IEEE LaTeX report draft with methods, references, result placeholders and limitations; no values/citations or completed work are fabricated.
- [x] Fill report metadata (Taimoor Ahmad, Computer Science, FAST NUCES, repository URL); measured validation and benchmark results, the eight-example caveat, and AI-use acknowledgement are in the draft.
- [x] Add a system workflow diagram and state that the step-wise training log was not retained, so learning curves cannot be reconstructed honestly.
- [x] Compile and visually review the IEEE report PDF (`report/genai_a2_report.pdf`). Correct its stale pending-PDF wording in LaTeX and recompile before final submission.
- [ ] Add a training-curve plot only if the original step-wise training log is recovered; do not invent one (logs were not retained).
- [x] Final test completed on Kaggle after enabling a T4 and removing incompatible `torchao` 0.10.0. Aggregate metrics are in the report; patient-level JSON must remain private.
- [ ] Capture screenshots and record 5–7 minute demo. **External**.
- [ ] Publish repository and model adapter download, submit via Google Classroom, and attend live evaluation. **External**.

