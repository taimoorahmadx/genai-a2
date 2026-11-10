# Assignment checklist

## Research and setup

- [ ] Inspect environment and hardware; identify free execution options and dependency constraints.
- [ ] Read primary papers and official model/dataset documentation; collect at least five credible sources.
- [ ] Compare at least two suitable open-weight autoregressive models; verify architecture, parameter count, license, tokenizer/context, and resource needs.
- [ ] Choose target and tokenizer-compatible draft model with evidence-based rationale.

## Data, training, and evaluation

- [ ] Inspect required dataset schema; implement clean preprocessing and explicit exclusion of prior model outputs.
- [ ] Generate and publish deterministic 37/8/8 patient-grouped split manifest; add leakage checks.
- [ ] Implement baseline inference and supervised fine-tuning; document objective, output schema, tokenization/truncation, configuration, hyperparameters, and checkpoint.
- [ ] Evaluate identical validation/test cases before and after tuning; keep test untouched until final evaluation.
- [ ] Implement criterion and evidence metrics, inclusion/exclusion breakdowns, confusion matrices, and documented empty-evidence handling.
- [ ] Analyze errors, contradictions, missing information, and unsupported evidence (at least eight distinct cases).

## Inference engineering

- [ ] Implement uncached reference and explicit KV-cache decoding; validate greedy token agreement at three or more lengths.
- [ ] Benchmark prefill/decode time, tokens per second, and peak memory with warm-up and repeated synchronized runs.
- [ ] Implement reference attention and tiled online-softmax attention; validate causal masks/numerical agreement and measure performance/memory.
- [ ] Implement greedy speculative decoding; validate equality with target greedy output and cache state; report acceptance and performance, including any slowdown.

## Application, report, and handoff

- [ ] Build browser-accessible screening workflow with criterion-level evidence, educational notice, export, and inference comparison.
- [ ] Add setup/reproduction instructions, environment specs, scripts, benchmark logs, and model download instructions.
- [ ] Write IEEE LaTeX report with interpreted results, figures/tables, limitations, citations, and AI-use appendix; generate PDF.
- [ ] Capture final demo screenshots/results and prepare the 5–7 minute walkthrough.
- [ ] Final review against every rubric item; make source project and artifacts ready for submission.

