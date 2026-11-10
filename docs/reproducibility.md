# Reproducibility and experiment log

## Environment inspected

- Linux x86_64, Python 3.10.12
- PyTorch 2.14.1 CPU-only; CUDA unavailable
- NumPy, scikit-learn, datasets, pytest, FastAPI and Streamlit importable at inspection time
- Transformers, PEFT, Accelerate absent at inspection time
- Dataset was prepared in Colab using the prescribed split: 37/8/8 patients and 603/206/206 annotations; leakage check passed. Manifest and processed notes remain in the Colab runtime, not this repository.
- A Colab Tesla T4 completed 3-epoch LoRA training (228 steps; train loss 0.1347; validation loss 0.04293). The adapter archive is present locally as ignored `trained-adapter.zip`; do not commit it. Colab-only changes selected FP16 on T4 and replaced `warmup_ratio` for Transformers 5 compatibility.
- Training loss values are not classification/evidence evaluation results.
- Same-split validation evaluations completed in Colab (206 annotations): baseline accuracy 0.2621, macro-F1 0.1109, evidence micro-F1 0.000; LoRA adapter accuracy 0.7233, macro-F1 0.3564, evidence micro-F1 0.4286. Inclusion: baseline accuracy/macro-F1 0.4242/0.1156, adapter 0.7273/0.2502 (66 rows). Exclusion: baseline 0.1857/0.0799, adapter 0.7214/0.1397 (140 rows). The adapter predicts `not excluded` for all exclusion examples and fails to recover rare `excluded` and `not applicable` categories. The final test command was attempted after freezing settings, but Colab refused GPU access because of usage limits before evaluation began; no test metrics were produced. Retry once GPU access returns.
- Manual error review of eight displayed adapter validation examples found one repeated failure pattern: each gold `not enough information` exclusion was predicted as `not excluded`. In multiple examples the evidence was irrelevant (demographics/comorbidities for unrelated criteria or travel for a discharge criterion); one pregnancy-related case omitted the annotated uterine-imaging evidence and cited symptoms instead. These eight criterion examples span only three patient notes, so they are not eight independent patient cases. This is a targeted qualitative sample, not a prevalence estimate. Full notes and row-level artifacts remain private in Colab and are not reproduced here.

## Commands

See the repository README for setup, dataset manifest creation, LoRA fine-tuning, validation, one-time final testing, attention checks, decoder benchmarks, and app startup.

## Results ledger

Colab training, validation, attention, and a reduced decoder smoke benchmark have run. The decoder timings are one-repeat exploratory measurements; a stable repeated benchmark, held-out test evaluation, and app demo remain pending. Keep generated data and raw patient notes out of version control.

| Experiment | Status | Artifact |
|---|---|---|
| Dataset schema inspection | Reviewed HF dataset card | linked in report refs |
| Fixed patient manifest | Generated and checked in Colab; manifest not copied to this repo yet | `data/splits/patient_split_seed42.json` |
| Train/validation LoRA | Completed in Colab T4; train loss 0.1347, validation loss 0.04293 | local ignored `trained-adapter.zip`; do not commit |
| Baseline/adapter validation | Completed in Colab; metrics above; row-level JSON remains in runtime and was not checked into Git | ignored `artifacts/results/` |
| Attention check/benchmark | Completed on Colab Tesla T4, CUDA; 2 warmups and 5 timed repeats per method; JSON remains in Colab | `artifacts/results/attention_benchmark.json` |
| KV/speculative decoder benchmark | Reduced smoke run completed on Colab T4; all token IDs matched, but only one timing repeat was used. Target/draft vocabulary and special token IDs verified equal. | `artifacts/results/decoding_benchmark.json` in Colab runtime |
| Browser demo | UI implemented; model-backed interaction not run. Streamlit startup smoke attempt was blocked when sandbox denied local socket bind (`PermissionError`). | external video pending |
| Final test | Attempted once after freezing settings; Colab GPU backend unavailable due to usage limits, so evaluation did not start and no results were produced. | rerun once when GPU access returns |

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

The first Colab decoder-benchmark attempt was interrupted (`KeyboardInterrupt`) during the initial uncached generation. After reducing the synthetic cases and using one repeat, a smoke benchmark completed on CUDA device `cuda:0` with Qwen2.5-1.5B-Instruct as target and Qwen2.5-0.5B-Instruct as draft. Target uncached, KV-cached, and speculative greedy outputs matched exactly at all tested lengths. A separate Colab check confirmed that target and draft tokenizers have identical vocabularies and special token IDs. The benchmark script now enforces this check before speculative decoding.

| Input/output tokens | Mode | Median seconds* | Tokens/s | Peak allocated memory (GiB) | Draft accepted/proposed |
|---|---|---:|---:|---:|---:|
| 32/4 | Uncached | 0.3682 | 10.86 | 3.815 | — |
| 32/4 | KV cached | 0.1986 | 20.14 | 3.816 | — |
| 32/4 | Speculative | 0.3097 | 12.92 | 3.825 | 4/4 |
| 64/4 | Uncached | 0.5572 | 7.18 | 3.821 | — |
| 64/4 | KV cached | 0.2052 | 19.49 | 3.823 | — |
| 64/4 | Speculative | 0.3242 | 12.34 | 3.841 | 4/4 |
| 128/8 | Uncached | 1.7165 | 4.66 | 3.845 | — |
| 128/8 | KV cached | 0.4228 | 18.92 | 3.849 | — |
| 128/8 | Speculative | 0.8797 | 9.09 | 3.893 | 8/8 |

*One measured repeat after one warmup; not a robust median estimate despite the output field name. KV cache was 1.85x, 2.72x, and 4.06x faster than uncached decoding at the three tested input lengths. Speculative decoding was slower than KV caching by 1.56x, 1.58x, and 2.08x, although it was faster than uncached in these runs. All draft tokens were accepted, but the generated token sequence was highly repetitive and outputs were very short. These synthetic smoke-test results do not establish typical speculative acceptance or production performance. The measured peak memory includes model allocations and rose from about 3.82 to 3.89 GiB; speculative mode had the highest peak. A longer multi-repeat benchmark with representative prompts and outputs is still needed.

