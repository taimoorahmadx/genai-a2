"""Prompt construction, model loading, constrained output parsing and prediction."""
from __future__ import annotations
import json, re
from typing import Any
from .data import LABELS, sentence_map

MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_PROMPT=("You classify one clinical trial criterion against one patient record. Use only the note and criterion. "
"Missing information is not proof of eligibility or ineligibility. Return only JSON with label (one of the allowed exact categories), "
"evidence_ids (integer IDs present in the note), and explanation based only on the record. Do not infer unsupported facts.")

def make_prompt(row: dict[str, Any]) -> str:
    return (f"{SYSTEM_PROMPT}\nAllowed labels: {', '.join(LABELS)}\nCriterion type: {row['criterion_type']}\n"
            f"Criterion: {row['criterion_text']}\nPatient note:\n{row['note']}\n"
            "JSON response: {\"label\": \"...\", \"evidence_ids\": [], \"explanation\": \"...\"}")

def parse_prediction(text: str, note: str) -> dict[str, Any]:
    try:
        start=text.find("{"); end=text.rfind("}"); obj=json.loads(text[start:end+1])
        label=str(obj.get("label","")).strip().lower(); ids=obj.get("evidence_ids",[]); explanation=str(obj.get("explanation",""))
    except (ValueError, TypeError, json.JSONDecodeError): label=""; ids=[]; explanation=""
    if label not in LABELS: label="not enough information"
    valid=set(sentence_map(note)); cleaned=[]
    for x in ids if isinstance(ids,list) else []:
        try: i=int(x)
        except (ValueError,TypeError): continue
        if i in valid and i not in cleaned: cleaned.append(i)
    sentences=sentence_map(note)
    return {"label":label,"evidence_ids":cleaned,"evidence_text":{str(i):sentences[i] for i in cleaned},"explanation":explanation}

def load_model(model_id: str=MODEL_ID, adapter: str|None=None):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc: raise RuntimeError("Install model extra with pip install -e '.[model]'") from exc
    import torch
    tok=AutoTokenizer.from_pretrained(model_id)
    model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype="auto",device_map="auto")
    if adapter:
        try: from peft import PeftModel
        except ImportError as exc: raise RuntimeError("Install PEFT to load the trained adapter") from exc
        model=PeftModel.from_pretrained(model,adapter)
    model.eval(); return tok,model

def predict(tokenizer: Any, model: Any, row: dict[str, Any], max_new_tokens: int=160) -> dict[str, Any]:
    import torch
    prompt=make_prompt(row)
    inp=tokenizer(prompt,return_tensors="pt",truncation=True,max_length=4096).to(model.device)
    with torch.inference_mode(): out=model.generate(**inp,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True)
    decoded=tokenizer.decode(out[0,inp.input_ids.shape[-1]:],skip_special_tokens=True)
    return parse_prediction(decoded,row["note"])

