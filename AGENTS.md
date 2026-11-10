# Project working notes

This repository is for GenAI Assignment 2, an individual research and implementation project. The detailed source of truth is `refrences/GenAI_Assignment#2.md`; use `docs/plan.md` and `docs/tasks.md` to resume work across sessions.

## Rules for this project

- The current setup task creates context only. Start implementation in a later chat when requested.
- Build and document the complete assignment: reproducible code, experiments, browser application, IEEE LaTeX report, and submission artifacts. Do not stop at a notebook or a plan.
- Use the required TrialGPT Criterion Annotations dataset for supervised training and the common comparison. Exclude its prior model predictions/explanations from inputs. Keep every patient's annotations together in the prescribed 37/8/8 split (sorted patient IDs, shuffled with `numpy.random.default_rng(42)`). Do not use the final test split for tuning.
- Preserve the dataset's label categories and sentence-ID evidence. Missing evidence is not proof of eligibility or ineligibility. Describe any overall trial verdict as an explicit aggregation, not a ground-truth label.
- Use an open-weight, locally executable autoregressive Transformer. Paid GPUs, APIs, and proprietary model services must not be required. Check model licensing, tokenizer compatibility, hardware needs, and freely accessible execution options before committing to a design.
- Implement and validate the cache-aware decoder, tiled online-softmax attention, and greedy speculative decoder; a library flag or API call alone does not satisfy the assignment.
- Never invent results, citations, completed work, or performance claims. Run experiments and report failures and limitations honestly. Keep the held-out test untouched until final evaluation.
- Keep patient data and credentials out of version control. Acknowledge AI assistance in the final submission.
- Separate work the assistant can complete in the repository from genuinely external actions (for example, recording a demo or submitting through Classroom). Do not ask the user to do repository work that can be done here.

