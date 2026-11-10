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

The Colab training, validation, and synthetic attention benchmark below are the only GPU experiments completed so far. No decoder hardware benchmark, held-out test evaluation, or app demo has run. Decoder equivalence on actual model weights is not claimed yet. Keep generated data and raw patient notes out of version control.

| Experiment | Status | Artifact |
|---|---|---|
| Dataset schema inspection | Reviewed HF dataset card | linked in report refs |
| Fixed patient manifest | Generated and checked in Colab; manifest not copied to this repo yet | `data/splits/patient_split_seed42.json` |
| Train/validation LoRA | Completed in Colab T4; train loss 0.1347, validation loss 0.04293; adapter zip downloaded by user | user-held zip; not in repo |
| Baseline/adapter validation | Completed in Colab; metrics above; row-level JSON remains in runtime and was not checked into Git | ignored `artifacts/results/` |
| Attention check/benchmark | Completed on Colab Tesla T4, CUDA; 2 warmups and 5 timed repeats per method; JSON remains in Colab | `artifacts/results/attention_benchmark.json` |
| KV/speculative decoder benchmark | Colab attempt interrupted before results; model dependencies and weights are cached in that runtime | `artifacts/results/decoding_benchmark.json` not created |
| Browser demo | UI implemented; model-backed interaction not run. Streamlit startup smoke attempt was blocked when sandbox denied local socket bind (`PermissionError`). | external video pending |

### Tiled attention benchmark (Colab Tesla T4)

The reference and tiled online-softmax outputs passed the script's `allclose` check in all six cases. Maximum absolute error was `5.3644e-7`; mean absolute errors ranged from `2.4397e-8` to `2.7728e-8`. Times below are median seconds over five repeats after two warmups. The tiled Python implementation was slower in every tested shape; the speed ratio is tiled/reference. Peak memory is PyTorch allocated bytes as measured by the script.

| Query/key length, head dim | Mask | Max abs. error | Reference ms | Tiled ms | Tiled/reference | Peak memory, ref / tiled MiB |
|---|---:|---:|---:|---:|---:|---:|
| 32/32, 64 | noncausal | 5.36e-7 | 0.179 | 0.784 | 4.38x | 8.34 / 8.44 |
| 32/32, 64 | causal | 3.58e-7 | 0.324 | 0.836 | 2.58x | 8.34 / 8.44 |
| 64/128, 64 | noncausal | 2.98e-7 | 0.202 | 4.983 | 24.68x | 8.87 / 8.75 |
| 64/128, 64 | causal | 2.98e-7 | 0.322 | 5.782 | 17.97x | 8.87 / 8.75 |
| 128/128, 128 | noncausal | 4.77e-7 | 0.190 | 9.889 | 51.98x | 10.12 / 9.97 |
| 128/128, 128 | causal | 3.58e-7 | 0.314 | 11.168 | 35.53x | 10.12 / 9.97 |

These are small synthetic tensors, not model workload measurements. The launch/Python overhead dominates at these sizes; the results establish numerical agreement but do not demonstrate a speed or meaningful memory benefit. At the smallest shape the tiled path used slightly more measured memory; at larger shapes its measured savings were about 0.12–0.15 MiB. Treat timings as a single Colab T4 run, not a general hardware claim.

The first Colab decoder-benchmark attempt was interrupted (`KeyboardInterrupt`) during the initial uncached generation, before any benchmark result was written. The target and draft models finished downloading and should be cached in that Colab runtime. No decoder timing or token-equivalence result is available yet.

