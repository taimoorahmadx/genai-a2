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
- [x] GPU LoRA training completed on Colab Tesla T4 for 3 epochs / 228 steps; reported train loss 0.1347 and best/final validation loss 0.04293. User downloaded adapter zip. Training log and artifact are not yet in repository; truncation still needs review.
- [x] Implement baseline/adapter evaluation on identical selected split, class/evidence metrics, inclusion/exclusion breakdowns and empty evidence policy.
- [x] Run baseline and adapter on the same 206 validation annotations. Baseline accuracy/macro-F1 0.262/0.111; adapter 0.723/0.356. Adapter evidence micro-F1 0.429 (baseline 0). Adapter is biased toward `not excluded`; per-class failure review remains required.
- [ ] One-time final test evaluation after settings are frozen; outputs will be ignored until reviewed and then selectively published.
- [x] Include per-example predictions and error fields in evaluation artifacts; review eight validation error examples. All eight gold `not enough information` labels were predicted as `not excluded`; several cited irrelevant or incomplete evidence. They came from only three patient notes, so these are eight criterion-level errors, not eight independent patient cases. Broader, patient-diverse error analysis remains advisable.

## Inference engineering

- [x] Implement uncached reference and explicit KV-cache greedy decoding; local mock verifies exact token equality at three output lengths.
- [ ] Verify against actual target model at three prompt/output lengths and benchmark synchronized repeated runs, prefill/decode latency, throughput and peak memory.
- [x] Implement reference scaled dot-product attention and tiled online-softmax attention.
- [x] CPU numerical tests cover three shapes and causal/noncausal masks; max/mean error computed by benchmark script.
- [x] Execute attention timing/memory experiment on Colab Tesla T4; all six cases passed equivalence checks (maximum absolute error 5.36e-7). Tiled attention was 2.58x–51.98x slower on these small shapes, with small/mixed peak-memory differences. Full measurements are in `docs/reproducibility.md`.
- [x] Implement greedy speculative decoding including target verification, correction after rejection, output/cache synchronization by re-prefill; local deterministic mock verifies mismatch correction.
- [x] Verify actual Qwen target, KV-cache, and speculative generated token IDs match at three reduced prompt/output lengths on Colab T4; record preliminary acceptance, speed, and memory results. **Pending:** run the explicit vocabulary check added to the benchmark script, then collect a longer repeated benchmark for stable performance estimates.

## Application, paper and handoff

- [x] Build standalone Streamlit screening workflow with evidence, missing information, explicit aggregation, educational notice and report export.
- [x] Build interactive model-backed inference comparison page; it measures uncached, cached and speculative paths and verifies output equality when run.
- [x] Add setup/reproduction instructions, optional dependencies, data handling and model download path.
- [x] Create IEEE LaTeX report draft with methods, references, result placeholders and limitations; no values/citations or completed work are fabricated.
- [ ] Populate report with real runs, analyzed figures/tables, author/repository details, eight-case analysis and AI-use acknowledgement; compile report PDF.
- [ ] Capture screenshots and record 5–7 minute demo. **External**.
- [ ] Publish repository and model adapter download, submit via Google Classroom, and attend live evaluation. **External**.

