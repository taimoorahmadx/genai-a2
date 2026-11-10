import json
import torch
from trial_screening.data import normalize_row, parse_ids, patient_split, make_manifest, sentence_map, FORBIDDEN_INPUT_FIELDS
from trial_screening.attention import reference_attention, tiled_online_attention
from trial_screening.metrics import evidence_report, classification_report
from trial_screening.modeling import parse_prediction
from trial_screening.decoding import greedy_cached, greedy_uncached, speculative_greedy
from types import SimpleNamespace

def row(i,p):
    return {"annotation_id":i,"patient_id":p,"note":"0. Age is 50. 1. Has disease.","criterion_type":"inclusion","criterion_text":"Age 50","expert_eligibility":"included","expert_sentences":"[0]","gpt4_explanation":"LEAK","gpt4_sentences":"[1]","gpt4_eligibility":"excluded","explanation_correctness":"wrong"}

def test_normalize_excludes_prior_model_fields():
    clean=normalize_row(row(1,"P"))
    assert not FORBIDDEN_INPUT_FIELDS.intersection(clean)
    assert clean["label"]=="included" and clean["evidence_ids"]==[0]

def test_parse_ids_and_sentences():
    assert parse_ids("[0, 2, 2]")==[0,2]
    assert sentence_map("0. Age is 50. 1. Has disease.")=={0:"Age is 50.",1:"Has disease."}

def test_patient_split_deterministic_and_disjoint():
    rows=[row(i,f"P{x}") for i,x in enumerate(range(53))]
    split=patient_split(rows); assert split==patient_split(rows)
    assert [len(split[k]) for k in ("train","validation","test")] == [37,8,8]
    assert make_manifest(rows,split)["leakage_check"]=="passed"

def test_tiled_attention_matches_reference_various_shapes():
    torch.manual_seed(2)
    for nq,nk,d in [(4,4,8),(7,11,5),(8,13,16)]:
        q=torch.randn(2,3,nq,d); k=torch.randn(2,3,nk,d); v=torch.randn(2,3,nk,d)
        for causal in (False,True):
            ref=reference_attention(q,k,v,causal)
            got=tiled_online_attention(q,k,v,causal,block_size=3)
            assert torch.allclose(ref,got,atol=1e-5,rtol=1e-4)

def test_prediction_rejects_unknown_evidence_ids():
    pred=parse_prediction('{"label":"included","evidence_ids":[0,8],"explanation":"supported"}',"0. Age is 50.")
    assert pred["evidence_ids"]==[0] and pred["evidence_text"]=={"0":"Age is 50."}

def test_metrics_keep_all_labels_and_document_empty_policy():
    report=classification_report(["included"],["not enough information"])
    assert len(report["per_class"])==6
    ev=evidence_report([[],[1]], [[],[]])
    assert "empty/empty" in ev["empty_policy"] and ev["micro_recall"]==0 and ev["macro_f1"]==0.5

class TinyCacheModel:
    """Deterministic test double whose KV cache stores prior input token IDs."""
    def __init__(self, step=1): self.step=step; self.config=SimpleNamespace(vocab_size=16)
    def __call__(self,input_ids,past_key_values=None,use_cache=False,**kwargs):
        prev=past_key_values[0][0][:,0,:,0].long() if past_key_values is not None else input_ids[:,:0]
        all_ids=torch.cat([prev,input_ids],dim=-1)
        logits=torch.full((input_ids.shape[0],input_ids.shape[1],16),-100.)
        for j in range(input_ids.shape[1]): logits[:,j,(all_ids[:,prev.shape[1]+j]+self.step)%16]=100.
        cache=((all_ids[:,None,:,None].float(),all_ids[:,None,:,None].float()),) if use_cache else None
        return SimpleNamespace(logits=logits,past_key_values=cache)

def test_cached_and_uncached_greedy_agree_across_lengths():
    model=TinyCacheModel(); ids=torch.tensor([[3,5,7,8,9],[2,4,6,8,10]])
    for in_len,out_len in ((1,1),(3,2),(5,5)):
        prompt=ids[:,:in_len]
        assert torch.equal(greedy_cached(model,prompt,out_len).token_ids,greedy_uncached(model,prompt,out_len).token_ids)

def test_speculative_greedy_corrects_draft_mismatches():
    ids=torch.tensor([[1,4,7]])
    target=TinyCacheModel(step=1); draft=TinyCacheModel(step=2)
    expected=greedy_uncached(target,ids,9).token_ids
    result=speculative_greedy(target,draft,ids,9,draft_tokens=3)
    assert torch.equal(result.token_ids,expected)
    assert result.proposed_draft_tokens>0

