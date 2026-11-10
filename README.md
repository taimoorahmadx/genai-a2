# Trial Criterion Screening

An educational, reproducible research prototype for criterion-level screening on the required TrialGPT Criterion Annotations benchmark. This is not a clinical decision tool. Predictions are generated locally by an open-weight model; the app does not silently substitute a heuristic when model weights are unavailable.

## Current state

The repository contains leakage-aware data preparation, LoRA training and evaluation, explicit uncached/KV-cached/speculative decoding, tiled online-softmax attention, a Streamlit browser app, interactive benchmarks, and an IEEE LaTeX report. Training and validation ran on a Colab T4; synthetic attention and reduced decoding benchmarks ran on Colab; the frozen held-out test ran once on Kaggle. The compiled report is `report/genai_a2_report.pdf`.

This is an educational research prototype, not a clinical decision system. Some submission evidence is still external or unavailable: the split manifest and benchmark JSON logs remain in hosted runtimes, the adapter is a local ignored file and has no published download link, the browser app has not been verified end to end, and no application screenshots or demo video are included. Step-wise training logs were not retained, so a training curve cannot be reconstructed. See `docs/reproducibility.md` for measured results and limitations; see the checklist below before submission.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[data,model,app,dev]'
```

The development environment inspected for this project was Linux x86_64, Python 3.10.12, PyTorch 2.14.1 CPU-only. GPU training results were obtained separately on Colab T4. A CPU-only environment is suitable for data prep and small algorithm checks, not evidence that fine-tuning is feasible there.

## Dataset and fixed split

```bash
python scripts/prepare_data.py
```

Downloads the official `ncbi/TrialGPT-Criterion-Annotations` HF dataset; writes `data/splits/patient_split_seed42.json`, train and validation JSONL under ignored `data/processed/`. The test patient IDs are recorded in the manifest, but test examples are not materialized by this preparation script. The preprocessing allowlists note, criterion type/text, expert label and expert evidence IDs. GPT-4 generated eligibility, evidence and explanation fields are never returned by the normalization layer. The manifest validates complete patient coverage and zero group overlap.

Patient IDs are sorted, shuffled using `numpy.random.default_rng(42)`, then split 37/8/8. The script intentionally fails if the official source no longer has the expected 53 distinct patients; inspect dataset changes before proceeding.

## Model and training

Default target: `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, 1.54B parameters, decoder-only, RoPE, GQA, 32,768 context). Draft candidate: `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0, 0.49B parameters). The Qwen2.5 family advertises a unified tokenizer; the benchmark and app compare vocabularies before speculative decoding. `microsoft/Phi-3.5-mini-instruct` is an alternative at 3.8B, MIT, and 128K context, with a higher weight/activation memory burden. The target's fp16 weights alone require roughly 3.1 GB before KV cache, activations and framework overhead; the 0.5B draft adds roughly 1 GB at fp16. LoRA training completed on a free Colab Tesla T4 in this project; actual GPU memory use is recorded with the benchmark outputs where available.

Training uses LoRA (rank 16, alpha 32, dropout .05) on Q/K/V/O projections, AdamW defaults from Transformers Trainer, 3 epochs, LR 2e-4, batch 1, gradient accumulation 8, 4096 token truncation, gradient checkpointing, and validation-loss checkpoint selection. The script uses FP16 on T4-class GPUs and BF16 only where supported; warmup is specified as an explicit step count for compatibility. Train/evaluation patient sets are distinct. Run:

```bash
python scripts/train_lora.py
```

The script uses answer-only loss and validation-loss checkpoint selection. A three-epoch Colab T4 run completed and the adapter was evaluated on validation and once on the held-out test after settings were frozen. Inspect truncation effects where logs/data permit; per-step training logs were not retained.

## Validation and final test

```bash
python scripts/evaluate.py --split validation
# Only once settings are frozen and final evaluation is intended:
python scripts/evaluate.py --split test --final-test
```

The evaluator reports all six expert categories, accuracy, macro-F1, class precision/recall/F1/confusion matrix, inclusion/exclusion breakdown, and evidence micro/macro scores. Empty evidence policy is recorded in output. Baseline and adapter runs use identical split examples. Never select settings using test results.

## Inference algorithm checks and benchmarks

```bash
pytest
python scripts/benchmark_attention.py
python scripts/benchmark_decoding.py
```

The attention implementation was numerically compared with the reference on six synthetic causal/noncausal cases on a Colab T4. A reduced decoder smoke benchmark verified output token equality at three lengths, but used one timing repeat; KV caching was fastest and speculative decoding did not beat it. The implementation re-prefills caches after verification rounds for correctness and may be slow. See `docs/reproducibility.md`. Result JSON files are not in the repository; keep any row-level evaluation outputs private because they contain patient notes.

## Browser application

```bash
cd /home/user/Desktop/genai-a2
source .venv/bin/activate
pip install -e '.[model,app]'
streamlit run app.py
```

Enter numbered patient sentences and one criterion per line (prefix `inclusion:` or `exclusion:`). The app loads the specified Hugging Face model and optional local adapter, displays category, explanation and evidence text, describes its transparent rule-based aggregation, and exports JSON. It requires network access for first-time model download unless weights are cached locally. The browser workflow has not yet been verified end to end with model weights. No fake/static prediction is shown when loading fails.

### Recording the application demo

Use a machine with enough RAM/GPU memory for Qwen2.5-1.5B. First-time model download requires internet and can take several minutes. To use the trained adapter, extract the adapter archive to a directory and enter that directory in the app's **Optional local LoRA adapter path** field. Do not upload patient data or adapter weights into GitHub.

Suggested 5–7 minute recording:

1. Show the repository and briefly point out the app, training/evaluation scripts, and report.
2. Run `streamlit run app.py`, open the local URL printed in the terminal, and show the educational-use and missing-information notices.
3. Enter a short synthetic numbered patient note and two criteria, one prefixed `inclusion:` and one `exclusion:`. Run screening; explain the displayed label, cited sentence IDs/text, and rule-based overall summary. Do not present the output as medical advice.
4. Download and open the JSON report to show the exported criterion results and notice.
5. Open **Inference benchmarks** from Streamlit's page navigation. Enter a short prompt and run the comparison; show token agreement, latency, throughput, memory availability, and speculative acceptance. This page loads both target and draft models and may exceed laptop resources; if it fails, report that honestly and show the measured smoke results in the report instead of claiming a live run.
6. Close by showing the report's limitations and the documented class-bias/slowdown findings.

Use only synthetic example text in the recording. Do not claim an end-to-end run succeeded until you have seen the predictions and downloaded the JSON yourself.

## Report and submission

The source is `report/main.tex`; the compiled three-page PDF is `report/genai_a2_report.pdf`. The paper includes author/repository details, a system workflow diagram, validation and held-out test results, inference results, error analysis, and limitations. The decoder timings are a one-repeat smoke benchmark, not a repeated robust benchmark. The report notes that the step-wise training log was not preserved, so no training curve is reported. The compiled report has been visually reviewed; its limitations text still lists the report PDF as pending and should be corrected in LaTeX before final submission. Publish an adapter download link, capture app screenshots, record the demo, and submit through Classroom externally.

## Submission checklist (excluding the required video)

- [x] Source code, dependency specification, data preparation, training/evaluation scripts, inference implementations, browser app source, and reproducibility instructions are in this repository.
- [x] IEEE LaTeX source and compiled PDF are present.
- [x] Training, validation, attention, reduced decoding, and one-time held-out test experiments were run and their available aggregate results are documented.
- [ ] Publish the fixed split manifest (patient IDs and counts) from the hosted runtime; do not include patient notes.
- [ ] Publish benchmark JSON logs or export safe aggregate logs from the hosted runtimes.
- [ ] Publish the trained adapter somewhere durable and add a working download link; do not commit model weights to Git.
- [ ] Verify the app end to end with an unseen input, evidence display, invalid input, JSON export, and benchmark page; capture screenshots.
- [ ] Correct the stale “report PDF ... remain pending” wording in `report/main.tex`, recompile, and replace the PDF.
- [ ] Submit the PDF and repository URL through Google Classroom and prepare for the live technical defense.

## AI-use acknowledgement

AI assistance was used for repository scaffolding and implementation support. The student must review, understand, validate, and accurately acknowledge all AI-assisted work in the final report.

