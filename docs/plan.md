# Assignment plan

## Goal and scope

Deliver a reproducible Transformer-based clinical-trial criterion screening system for Assignment 2. It must predict criterion-level eligibility categories from patient text, cite supporting patient sentence IDs, implement the required inference methods, expose them through a browser application, and include a research report and submission evidence. This file captures the work plan; implementation is for a later chat.

## Non-negotiable evaluation protocol

- Dataset: `ncbi/TrialGPT-Criterion-Annotations`, the required supervised benchmark.
- Split by unique patient: sort IDs, shuffle with NumPy `default_rng(42)`, then assign 37 train, 8 validation, and 8 test patients. Save a manifest and verify no patient leakage.
- Never feed dataset-provided model predictions or explanations to the system. Do not tune on test cases.
- Preserve all applicable expert label categories, including insufficient-information and not-applicable outcomes. Retain criterion type and evidence sentence IDs.
- Report classification accuracy, macro-F1, per-class precision/recall/F1 and confusion matrices overall and by inclusion/exclusion. Report evidence precision/recall/F1 and state how empty gold/predicted evidence sets are scored.
- Report overall patient–trial outcomes only as transparent aggregation rules; the dataset does not provide those as separately labeled outcomes.

## Workstreams and deliverables

1. **Research and model choice:** review at least five credible sources; compare at least two open-weight decoder candidates on architecture, parameters, license, tokenizer/context, and hardware feasibility. Select only after verifying current documentation and available compute. Record alternatives and reasons for decisions.
2. **Data and fine-tuning:** create deterministic preprocessing and split manifest; define structured label/evidence output; train on train patients, select settings on validation, and reserve test for final evaluation. Compare the same base model before and after fine-tuning. Save configs, logs/curves, and distributable model adapter/weights or a documented download path.
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

## Completion boundary

Repository work should be completed by the assistant. The user may need to perform only external actions unavailable here: operate a free GPU session if required and interactive access cannot be automated, record the required demo video, submit through Google Classroom, and attend the live evaluation. Reassess GPU needs after model/hardware selection rather than assuming them now.

