"""Criterion label and sentence evidence metrics."""
from __future__ import annotations
from typing import Any
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from .data import LABELS

def classification_report(gold: list[str], pred: list[str]) -> dict[str, Any]:
    labels=list(LABELS); p,r,f,s=precision_recall_fscore_support(gold,pred,labels=labels,zero_division=0)
    return {"n":len(gold),"accuracy":float(accuracy_score(gold,pred)),"macro_f1":float(sum(f)/len(labels)),
        "per_class":{lab:{"precision":float(p[i]),"recall":float(r[i]),"f1":float(f[i]),"support":int(s[i])} for i,lab in enumerate(labels)},
        "confusion_matrix":{"labels":labels,"counts":confusion_matrix(gold,pred,labels=labels).tolist()}}

def evidence_report(gold: list[list[int]], pred: list[list[int]]) -> dict[str, Any]:
    tp=fp=fn=0; exact=[]
    for g,p in zip(gold,pred):
        gs,ps=set(g),set(p); tp+=len(gs&ps); fp+=len(ps-gs); fn+=len(gs-ps)
        precision=len(gs&ps)/len(ps) if ps else (1. if not gs else 0.)
        recall=len(gs&ps)/len(gs) if gs else (1. if not ps else 0.)
        exact.append((precision,recall,2*precision*recall/(precision+recall) if precision+recall else 0.))
    precision=tp/(tp+fp) if tp+fp else 1.; recall=tp/(tp+fn) if tp+fn else 1.
    return {"micro_precision":precision,"micro_recall":recall,"micro_f1":2*precision*recall/(precision+recall) if precision+recall else 0.,
            "macro_precision":sum(x[0] for x in exact)/len(exact) if exact else 0.,"macro_recall":sum(x[1] for x in exact)/len(exact) if exact else 0.,
            "macro_f1":sum(x[2] for x in exact)/len(exact) if exact else 0.,
            "empty_policy":"Per-record empty/empty scores 1 for precision and recall; one-sided empty scores 0. Micro empty/empty corpus scores 1."}

def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result={"overall":classification_report([x["gold_label"] for x in records],[x["pred_label"] for x in records]),
            "evidence":evidence_report([x["gold_evidence_ids"] for x in records],[x["pred_evidence_ids"] for x in records])}
    for kind in ("inclusion","exclusion"):
        subset=[x for x in records if x["criterion_type"]==kind]
        result[kind]=classification_report([x["gold_label"] for x in subset],[x["pred_label"] for x in subset]) if subset else None
    return result

