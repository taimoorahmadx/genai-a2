# Assignment plan

## Goal and scope

Deliver a reproducible Transformer-based clinical-trial criterion screening system for Assignment 2. It must predict criterion-level eligibility categories from patient text, cite supporting patient sentence IDs, implement the required inference methods, expose them through a browser application, and include a research report and submission evidence. Implementation is underway at the user's request; current status is tracked in `docs/tasks.md` and `docs/reproducibility.md`.

## Non-negotiable evaluation protocol

- Dataset: `ncbi/TrialGPT-Criterion-Annotations`, the required supervised benchmark.
- Split by unique patient: sort IDs, shuffle with NumPy `default_rng(42)`, then assign 37 train, 8 validation, and 8 test patients. Save a manifest and verify no patient leakage.
- Never feed dataset-provided model predictions or explanations to the system. Do not tune on test cases.
- Preserve all applicable expert label categories, including insufficient-information and not-applicable outcomes. Retain criterion type and evidence sentence IDs.
- Report classification accuracy, macro-F1, per-class precision/recall/F1 and confusion matrices overall and by inclusion/exclusion. Report evidence precision/recall/F1 and state how empty gold/predicted evidence sets are scored.
- Report overall patient–trial outcomes only as transparent aggregation rules; the dataset does not provide those as separately labeled outcomes.

## Workstreams and deliverables

1. **Research and model choice:** review at least five credible sources; compare at least two open-weight decoder candidates on architecture, parameters, license, tokenizer/context, and hardware feasibility. Select only after verifying current documentation and available compute. Record alternatives and reasons for decisions.
2. **Data and fine-tuning:** preprocessing, split generation, answer-only LoRA training and evaluation code exist. The selected working pair is Qwen2.5-1.5B-Instruct (target) and Qwen2.5-0.5B-Instruct (draft), both Apache-2.0 with the Qwen2.5 family tokenizer. This remains provisional until downloaded and validated on the free accelerator. Train on train patients, select on validation, reserve test for final evaluation.
3. **Criterion and evidence evaluation:** build a traceable prediction pipeline; evaluate all required metrics on validation and final test; inspect representative errors, contradictory cases, and at least eight distinct failures, including missing-information and unsupported-evidence cases.
4. **Inference methods:** implement a no-reuse reference and explicit KV-cache reuse/update; verify greedy token agreement across at least three lengths and benchmark synchronized repeated runs (prefill, decode, tokens/s, peak memory). Implement reference scaled dot-product attention and tiled online-softmax attention; test causal masking, numerical error, speed, and memory. Implement greedy speculative decoding with a compatible tokenizer; verify token equality, cache synchronization, acceptance rate, latency, throughput, and memory. Explain slowdowns when observed.
5. **Application:** provide a standalone browser-accessible workflow for patient text and trial criteria, per-criterion labels/evidence/missing information, readable export, educational-use notice, and a live inference comparison/benchmark view. Document startup; users must not need a notebook or IDE to use it.
6. **Paper and submission:** write an IEEE-format LaTeX paper with methods, interpreted figures/tables, results, limitations, and references. Include reproducibility instructions, source repository URL, model download, split manifest, benchmark logs, AI-use appendix, report PDF, and a 5–7 minute application demo link.

## Decisions to make during implementation

- Inspect the dataset schema and library versions before fixing preprocessing or metric edge-case behavior.
- Research and compare model candidates, then choose a model/draft pair that fit freely available hardware and have compatible tokenization for speculative decoding.
- Measure actual hardware before setting training and benchmark configurations. Use free Colab/Kaggle or university resources if local resources are insufficient; no paid service should be required. Mark any accelerator-dependent runs clearly.
- Choose a practical application and packaging stack after inspecting the environment. Keep the UI functional without fake prediction results.
- Verify the rubric's stated total: its section allocations add to 135 although the heading says 100. Report the assignment's rubric as written and avoid silently normalizing marks.

## Current implementation boundary

Implemented: preprocessing/split code, LoRA training and evaluation scripts, attention and decoder algorithms, a Streamlit app with interactive benchmark page, and an IEEE report draft. Dataset preparation, LoRA training, validation evaluation, synthetic T4 attention benchmarking, and a reduced T4 decoder smoke benchmark have completed. The final test was attempted after freezing settings but could not start because Colab GPU access was quota-limited. The report PDF, app-backed demo, and external submission remain unfinished. See `docs/tasks.md` and `todo.md` for the current checklist.

The inspected local runtime is CPU-only (PyTorch 2.14.1); Transformers/PEFT are not installed. GPU-dependent final evaluation and full model-backed app/benchmark checks need Colab or university GPU access. No paid service is required. Colab access is currently blocked by usage limits, so use the waiting period for report metadata and demo preparation.

## Completion boundary

Repository work should be completed by the assistant. The user may need to perform only external actions unavailable here: operate a free GPU session if required and interactive access cannot be automated, record the required demo video, submit through Google Classroom, and attend the live evaluation. Reassess GPU needs after model/hardware selection rather than assuming them now.

