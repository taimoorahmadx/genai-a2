# Reproducibility and experiment log

## Environment inspected

- Linux x86_64, Python 3.10.12
- PyTorch 2.14.1 CPU-only; CUDA unavailable
- NumPy, scikit-learn, datasets, pytest, FastAPI and Streamlit importable at inspection time
- Transformers, PEFT, Accelerate absent at inspection time
- Dataset was prepared in Colab using the prescribed split: 37/8/8 patients and 603/206/206 annotations; leakage check passed. Manifest and processed notes remain in the Colab runtime, not this repository.
- A Colab Tesla T4 completed 3-epoch LoRA training (228 steps; train loss 0.1347; validation loss 0.04293). The user downloaded the adapter zip; it has not been independently inspected or added here. Colab-only changes selected FP16 on T4 and replaced `warmup_ratio` for Transformers 5 compatibility.
- Training loss values are not classification/evidence evaluation results.
- Same-split validation evaluations completed in Colab (206 annotations): baseline accuracy 0.2621, macro-F1 0.1109, evidence micro-F1 0.000; LoRA adapter accuracy 0.7233, macro-F1 0.3564, evidence micro-F1 0.4286. Inclusion: baseline accuracy/macro-F1 0.4242/0.1156, adapter 0.7273/0.2502 (66 rows). Exclusion: baseline 0.1857/0.0799, adapter 0.7214/0.1397 (140 rows). The adapter's high accuracy is dominated by the `not excluded` category (101/140 gold exclusion cases); it predicts that category for every exclusion example and fails to recover the rare `excluded` and `not applicable` categories. The test split has not been evaluated.
- Manual error review of eight displayed adapter validation examples found one repeated failure pattern: each gold `not enough information` exclusion was predicted as `not excluded`. In multiple examples the evidence was irrelevant (demographics/comorbidities for unrelated criteria or travel for a discharge criterion); one pregnancy-related case omitted the annotated uterine-imaging evidence and cited symptoms instead. These eight criterion examples span only three patient notes, so they are not eight independent patient cases. This is a targeted qualitative sample, not a prevalence estimate. Full notes and row-level artifacts remain private in Colab and are not reproduced here.

## Commands

See the repository README for setup, dataset manifest creation, LoRA fine-tuning, validation, one-time final testing, attention checks, decoder benchmarks, and app startup.

## Results ledger

The Colab training and validation runs below are the only model experiments completed so far. No attention/decoder hardware benchmark, held-out test evaluation, or app demo has run. Attention correctness has local CPU unit coverage; decoder equivalence on actual model weights is not claimed yet. Keep generated data and raw patient notes out of version control.

| Experiment | Status | Artifact |
|---|---|---|
| Dataset schema inspection | Reviewed HF dataset card | linked in report refs |
| Fixed patient manifest | Generated and checked in Colab; manifest not copied to this repo yet | `data/splits/patient_split_seed42.json` |
| Train/validation LoRA | Completed in Colab T4; train loss 0.1347, validation loss 0.04293; adapter zip downloaded by user | user-held zip; not in repo |
| Baseline/adapter validation | Completed in Colab; metrics above; row-level JSON remains in runtime and was not checked into Git | ignored `artifacts/results/` |
| Attention check/benchmark | Unit test available; timings not run | ignored `artifacts/results/` |
| KV/speculative decoder benchmark | Not run; model dependency/weights absent | ignored `artifacts/results/` |
| Browser demo | UI implemented; model-backed interaction not run. Streamlit startup smoke attempt was blocked when sandbox denied local socket bind (`PermissionError`). | external video pending |

