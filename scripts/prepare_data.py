#!/usr/bin/env python3
"""Download official TrialGPT annotations and create the fixed split manifest."""
import json
from pathlib import Path
from trial_screening.data import load_dataset, make_manifest, patient_split, summary, write_manifest

def main():
    rows=load_dataset(); split=patient_split(rows); manifest=make_manifest(rows,split)
    write_manifest("data/splits/patient_split_seed42.json",manifest)
    # Store permitted fields only; partition files keep groups intact.
    for name, ids in split.items():
        part=[r for r in rows if r["patient_id"] in set(ids)]
        if name=="test": continue  # do not materialize test examples during preparation
        out=Path("data/processed")/f"{name}.jsonl"; out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in part))
    print(json.dumps({"summary":summary(rows),"manifest":manifest},indent=2))
if __name__=="__main__": main()

