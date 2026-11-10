# Trial Criterion Screening

An educational, reproducible research prototype for criterion-level screening on the required TrialGPT Criterion Annotations benchmark. This is not a clinical decision tool. Predictions are generated locally by an open-weight model; the app does not silently substitute a heuristic when model weights are unavailable.

## Current state

The repository contains data handling, the deterministic split generator, LoRA training and evaluation, cache-aware and uncached decoding, online-softmax attention, speculative decoding, a browser app, benchmarks, and an IEEE LaTeX report draft. Colab T4 training and validation, synthetic attention and reduced decoder benchmarks, and the one-time held-out test on Kaggle have run. The adapter archive is present locally as an ignored file; never commit it or patient-level outputs. See `docs/reproducibility.md` for measured results and limitations.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[data,model,app,dev]'
```

The inspected development environment was Linux x86_64, Python 3.10.12, PyTorch 2.14.1 CPU-only, NumPy, scikit-learn, FastAPI and Streamlit available. Transformers, PEFT and Accelerate were absent. This CPU-only environment is useful for data prep and algorithm checks; it is not treated as evidence of feasible 0.5B fine-tuning.

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

The attention implementation is CPU-tested and has a Colab T4 synthetic benchmark. A reduced T4 decoder smoke benchmark verified output token equality at three lengths, but used one timing repeat; cached decoding was fastest and speculative decoding did not beat it. The benchmark now checks target/draft vocabulary equality; run it before the next speculative experiment. The implementation re-prefills caches after verification rounds for correctness and may be slow. See `docs/reproducibility.md`; generated result files are ignored because they may contain sensitive data.

## Browser application

```bash
streamlit run app.py
```

Enter numbered patient sentences and one criterion per line (prefix `inclusion:` or `exclusion:`). The app loads the specified local/open Hugging Face model, displays category, explanation and evidence text, describes its transparent heuristic aggregation, and exports JSON. It requires network access for first-time model download unless weights are cached locally. No fake/static prediction is shown when loading fails.

## Report and submission

See `report/main.tex` and `docs/reproducibility.md`. The report draft includes author/repository details, a system workflow diagram, validation and held-out test results, inference results, error analysis, and limitations. A repeated decoder benchmark, compiled PDF review, adapter distribution, and app demo remain pending. The step-wise training log was not preserved, so no training curve is reported. Record the demo and submit through Classroom externally.

## AI-use acknowledgement

AI assistance was used for repository scaffolding and implementation support. The student must review, understand, validate, and accurately acknowledge all AI-assisted work in the final report.

