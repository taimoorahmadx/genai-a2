# Reproducibility and experiment log

## Environment inspected

- Linux x86_64, Python 3.10.12
- PyTorch 2.14.1 CPU-only; CUDA unavailable
- NumPy, scikit-learn, datasets, pytest, FastAPI and Streamlit importable at inspection time
- Transformers, PEFT, Accelerate absent at inspection time
- No patient dataset was present locally; no data or weights were downloaded

## Commands

See the repository README for setup, dataset manifest creation, LoRA fine-tuning, validation, one-time final testing, attention checks, decoder benchmarks, and app startup.

## Results ledger

No model training, benchmark, held-out evaluation, or app demo has yet run. Consequently there are no performance values to report. Attention correctness is tested by the repository unit tests; decoder equivalence requires the optional model dependencies and model weights and is not claimed yet. Keep generated data and raw patient notes out of version control.

| Experiment | Status | Artifact |
|---|---|---|
| Dataset schema inspection | Reviewed HF dataset card | linked in report refs |
| Fixed patient manifest | Script implemented, not generated (dataset download unavailable/not run) | `data/splits/patient_split_seed42.json` |
| Train/validation LoRA | Not run; GPU needed for practical execution | ignored `artifacts/checkpoints/` |
| Baseline/adapter evaluation | Not run | ignored `artifacts/results/` |
| Attention check/benchmark | Unit test available; timings not run | ignored `artifacts/results/` |
| KV/speculative decoder benchmark | Not run; model dependency/weights absent | ignored `artifacts/results/` |
| Browser demo | UI implemented; model-backed interaction not run. Streamlit startup smoke attempt was blocked when sandbox denied local socket bind (`PermissionError`). | external video pending |

