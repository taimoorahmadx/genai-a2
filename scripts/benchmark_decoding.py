#!/usr/bin/env python3
"""Warm, synchronize and benchmark no-cache, explicit KV-cache and speculative generation."""
import argparse,json,time
from pathlib import Path
import torch
from trial_screening.modeling import load_model,MODEL_ID
from trial_screening.decoding import greedy_uncached,greedy_cached,speculative_greedy

def sync(device):
    if device.type=="cuda": torch.cuda.synchronize(device)
def run(fn,device,repeats):
    for _ in range(1): fn()
    vals=[]
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        sync(device); t=time.perf_counter(); result=fn(); sync(device); vals.append(time.perf_counter()-t)
    return result,sorted(vals)[len(vals)//2],torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None
def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",default=MODEL_ID); p.add_argument("--draft",default="Qwen/Qwen2.5-0.5B-Instruct"); p.add_argument("--repeats",type=int,default=5); a=p.parse_args()
    tok,target=load_model(a.model); _,draft=load_model(a.draft); device=next(target.parameters()).device
    data=[]
    for in_len,out_len in [(32,16),(128,32),(512,64)]:
        ids=tok("A patient record and criterion. "+"clinical detail "*in_len,return_tensors="pt",truncation=True,max_length=in_len).input_ids.to(device)
        reference=greedy_uncached(target,ids,out_len).token_ids
        for name,fn in [("uncached",lambda:greedy_uncached(target,ids,out_len)),("cached",lambda:greedy_cached(target,ids,out_len)),("speculative",lambda:speculative_greedy(target,draft,ids,out_len))]:
            res,seconds,peak=run(fn,device,a.repeats)
            data.append({"input_tokens":ids.shape[-1],"requested_output_tokens":out_len,"mode":name,"median_seconds":seconds,"tokens_per_second":out_len/seconds,"peak_memory_bytes":peak,"greedy_tokens_match":torch.equal(reference,res.token_ids),"token_ids":res.token_ids[0].tolist(),"accepted_draft_tokens":res.accepted_draft_tokens,"proposed_draft_tokens":res.proposed_draft_tokens})
    Path("artifacts/results").mkdir(parents=True,exist_ok=True); Path("artifacts/results/decoding_benchmark.json").write_text(json.dumps({"model":a.model,"draft":a.draft,"device":str(device),"warmups":1,"repeats":a.repeats,"runs":data},indent=2))
if __name__=="__main__": main()

