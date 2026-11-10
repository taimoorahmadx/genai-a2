# Trial Criterion Screening

An educational, reproducible research prototype for criterion-level screening on the required TrialGPT Criterion Annotations benchmark. This is not a clinical decision tool. Predictions are generated locally by an open-weight model; the app does not silently substitute a heuristic when model weights are unavailable.

## Current state

The repository now contains data handling, the required deterministic split generator, a LoRA training path, criterion/evidence evaluation, cache-aware and uncached greedy decoding, online-softmax attention, speculative decoding, a browser app, benchmarks, and an IEEE LaTeX report draft. No model download, GPU training, benchmark, test-set evaluation, application recording, or final submission has been performed. Results are not claimed until corresponding scripts are run and artifacts reviewed.

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

Default target: `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, 1.54B parameters, decoder-only, RoPE, GQA, 32,768 context). Draft candidate: `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0, 0.49B parameters). The Qwen2.5 family advertises a unified tokenizer; the benchmark and app compare vocabularies before speculative decoding. `microsoft/Phi-3.5-mini-instruct` is an alternative at 3.8B, MIT, and 128K context, with a higher weight/activation memory burden. The target's fp16 weights alone require roughly 3.1 GB before KV cache, activations and framework overhead; the 0.5B draft adds roughly 1 GB at fp16. GPU LoRA feasibility must be measured on the free environment actually used.

Training uses LoRA (rank 16, alpha 32, dropout .05) on Q/K/V/O projections, AdamW defaults from Transformers Trainer, 3 epochs, LR 2e-4, batch 1, gradient accumulation 8, 4096 token truncation, gradient checkpointing, and validation-loss checkpoint selection. Train/evaluation patient sets are distinct. Run:

```bash
python scripts/train_lora.py
```

The current script is a reproducibility scaffold. Its default `Trainer` causal-LM loss includes prompt tokens as well as answer tokens; improve to answer-only masking and inspect truncation before treating this config as a final experiment. GPU feasibility, training curves, adapter artifact and performance remain unverified. Do not call this model fine-tuned until the run completes and is evaluated.

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

The attention implementation is independently tested on CPU. Decoder benchmarks require downloading a supported HF model. Reports include warmup, repeat count, synchronized timing on CUDA, output token IDs, throughput, and CUDA peak memory when available. Greedy speculative decoding assumes identical tokenizers; verify target/draft token equality. It re-prefills caches after verification rounds for correctness and may therefore be slow. Benchmark output is ignored from git until reviewed; publish real logs when available.

## Browser application

```bash
streamlit run app.py
```

Enter numbered patient sentences and one criterion per line (prefix `inclusion:` or `exclusion:`). The app loads the specified local/open Hugging Face model, displays category, explanation and evidence text, describes its transparent heuristic aggregation, and exports JSON. It requires network access for first-time model download unless weights are cached locally. No fake/static prediction is shown when loading fails.

## Report and submission

See `report/main.tex` and `docs/reproducibility.md`. The report is a draft until real results, figures, failure analysis (at least eight distinct reviewed cases), author details, repository URL, adapter distribution, PDF, and demo are supplied. The rubric totals 135 despite its 100-mark heading; it is reported as printed in the assignment. Record the demo and submit through Classroom externally.

## AI-use acknowledgement

AI assistance was used for repository scaffolding and implementation support. The student must review, understand, validate, and accurately acknowledge all AI-assisted work in the final report.

