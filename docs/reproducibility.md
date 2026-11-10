# Reproducibility and experiment log

## Environment inspected

- Linux x86_64, Python 3.10.12
- PyTorch 2.14.1 CPU-only; CUDA unavailable
- NumPy, scikit-learn, datasets, pytest, FastAPI and Streamlit importable at inspection time
- Transformers, PEFT, Accelerate absent at inspection time
- Dataset was prepared in Colab using the prescribed split: 37/8/8 patients and 603/206/206 annotations; leakage check passed. Manifest and processed notes remain in the Colab runtime, not this repository.
- A Colab Tesla T4 completed 3-epoch LoRA training (228 steps; train loss 0.1347; validation loss 0.04293). The user downloaded the adapter zip; it has not been independently inspected or added here. Colab-only changes selected FP16 on T4 and replaced `warmup_ratio` for Transformers 5 compatibility.
- Training loss values are not classification/evidence evaluation results.

## Commands

See the repository README for setup, dataset manifest creation, LoRA fine-tuning, validation, one-time final testing, attention checks, decoder benchmarks, and app startup.

## Results ledger

No model training, benchmark, held-out evaluation, or app demo has yet run. Consequently there are no performance values to report. Attention correctness is tested by the repository unit tests; decoder equivalence requires the optional model dependencies and model weights and is not claimed yet. Keep generated data and raw patient notes out of version control.

| Experiment | Status | Artifact |
|---|---|---|
| Dataset schema inspection | Reviewed HF dataset card | linked in report refs |
| Fixed patient manifest | Generated and checked in Colab; manifest not copied to this repo yet | `data/splits/patient_split_seed42.json` |
| Train/validation LoRA | Completed in Colab T4; train loss 0.1347, validation loss 0.04293; adapter zip downloaded by user | user-held zip; not in repo |
| Baseline/adapter evaluation | Not run | ignored `artifacts/results/` |
| Attention check/benchmark | Unit test available; timings not run | ignored `artifacts/results/` |
| KV/speculative decoder benchmark | Not run; model dependency/weights absent | ignored `artifacts/results/` |
| Browser demo | UI implemented; model-backed interaction not run. Streamlit startup smoke attempt was blocked when sandbox denied local socket bind (`PermissionError`). | external video pending |

