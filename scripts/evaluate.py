#!/usr/bin/env python3
"""Evaluate predictions against a split; test access requires explicit final-eval flag."""
import argparse, json
from pathlib import Path
from trial_screening.data import load_dataset, sentence_map
from trial_screening.metrics import evaluate
from trial_screening.modeling import load_model,predict

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--split",choices=["validation","test"],default="validation"); ap.add_argument("--final-test",action="store_true"); ap.add_argument("--adapter"); ap.add_argument("--base-model"); a=ap.parse_args()
    if a.split=="test" and not a.final_test: ap.error("Test evaluation requires --final-test after all tuning is frozen")
    manifest=json.loads(Path("data/splits/patient_split_seed42.json").read_text()); ids=set(manifest["patients"][a.split])
    rows=[r for r in load_dataset() if r["patient_id"] in ids]
    tok,model=load_model(a.base_model or __import__('trial_screening.modeling',fromlist=['MODEL_ID']).MODEL_ID,a.adapter)
    records=[]
    for r in rows:
        p=predict(tok,model,r)
        records.append({"criterion_type":r["criterion_type"],"gold_label":r["label"],"pred_label":p["label"],"gold_evidence_ids":r["evidence_ids"],"pred_evidence_ids":p["evidence_ids"],"annotation_id":r["annotation_id"],"patient_id":r["patient_id"],"criterion_text":r["criterion_text"],"note":r["note"],"explanation":p["explanation"]})
    result={"split":a.split,"model":a.base_model,"adapter":a.adapter,"metrics":evaluate(records),"predictions":records}
    out=Path("artifacts/results")/f"{a.split}_{'finetuned' if a.adapter else 'baseline'}.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)); print(json.dumps(result["metrics"],indent=2))
if __name__=="__main__": main()

