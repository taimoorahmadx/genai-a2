"""Dataset loading, leakage-safe preprocessing and patient grouped splitting."""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
import numpy as np

LABELS = ("included", "not included", "not enough information", "excluded", "not excluded", "not applicable")
FORBIDDEN_INPUT_FIELDS = {"gpt4_explanation", "gpt4_sentences", "gpt4_eligibility", "explanation_correctness"}

def parse_ids(value: Any) -> list[int]:
    if value is None: return []
    if isinstance(value, (list, tuple)): raw = value
    else:
        try: raw = json.loads(value)
        except (TypeError, json.JSONDecodeError): raw = re.findall(r"\d+", str(value))
    out = []
    for x in raw:
        try: out.append(int(x))
        except (ValueError, TypeError): pass
    return sorted(set(out))

def sentence_map(note: str) -> dict[int, str]:
    # TrialGPT note sentence markers follow the start of the note or sentence
    # punctuation. Requiring that boundary avoids mistaking "age 50. ..." for ID 50.
    matches = list(re.finditer(r"(?:^|(?<=[.!?]\s))(\d+)\.\s+", note))
    return {int(m.group(1)): note[m.end():matches[i+1].start() if i+1 < len(matches) else len(note)].strip() for i, m in enumerate(matches)}

def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    label = str(row.get("expert_eligibility", "")).strip().lower()
    if label not in LABELS: raise ValueError(f"Unexpected expert label: {label!r}")
    kind = str(row.get("criterion_type", "")).strip().lower()
    if kind not in {"inclusion", "exclusion"}: raise ValueError(f"Unexpected criterion type: {kind!r}")
    return {"annotation_id": int(row["annotation_id"]), "patient_id": str(row["patient_id"]), "note": str(row["note"]),
            "trial_id": str(row.get("trial_id", "")), "trial_title": str(row.get("trial_title", "")),
            "criterion_type": kind, "criterion_text": str(row["criterion_text"]), "label": label,
            "evidence_ids": parse_ids(row.get("expert_sentences"))}

def load_dataset() -> list[dict[str, Any]]:
    try: from datasets import load_dataset
    except ImportError as exc: raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc
    ds = load_dataset("ncbi/TrialGPT-Criterion-Annotations", split="train")
    return [normalize_row(r) for r in ds]

def patient_split(rows: Iterable[dict[str, Any]], seed: int = 42) -> dict[str, list[str]]:
    ids = sorted({str(r["patient_id"]) for r in rows})
    if len(ids) != 53: raise ValueError(f"Expected 53 unique patients for prescribed split; found {len(ids)}")
    shuffled = np.random.default_rng(seed).permutation(ids).tolist()
    return {"train": shuffled[:37], "validation": shuffled[37:45], "test": shuffled[45:]}

def make_manifest(rows: list[dict[str, Any]], split: dict[str, list[str]]) -> dict[str, Any]:
    groups = {k: set(v) for k, v in split.items()}
    union = set.union(*groups.values())
    if union != {r["patient_id"] for r in rows}: raise ValueError("Manifest does not cover dataset patients")
    if sum(map(len, groups.values())) != len(union): raise ValueError("Patient leakage across splits")
    return {"dataset": "ncbi/TrialGPT-Criterion-Annotations", "seed": 42,
            "algorithm": "sort unique patient IDs; numpy.random.default_rng(42).permutation; assign 37/8/8",
            "patient_counts": {k: len(v) for k,v in groups.items()},
            "annotation_counts": {k: sum(r["patient_id"] in v for r in rows) for k,v in groups.items()},
            "patients": split, "leakage_check": "passed", "excluded_fields": sorted(FORBIDDEN_INPUT_FIELDS)}

def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(manifest, indent=2)+"\n")

def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"rows": len(rows), "patients": len({r["patient_id"] for r in rows}),
            "labels": dict(Counter(r["label"] for r in rows)),
            "criterion_types": dict(Counter(r["criterion_type"] for r in rows))}

