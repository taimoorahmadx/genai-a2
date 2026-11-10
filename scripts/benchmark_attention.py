#!/usr/bin/env python3
"""Numerical and runtime comparison of reference and tiled attention."""
import json,time
from pathlib import Path
import torch
from trial_screening.attention import reference_attention,tiled_online_attention

def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"; rows=[]
    for qlen,klen,d in [(32,32,64),(64,128,64),(128,128,128)]:
        torch.manual_seed(42); q=torch.randn(1,4,qlen,d,device=dev); k=torch.randn(1,4,klen,d,device=dev); v=torch.randn_like(k)
        for causal in (False,True):
            ref=reference_attention(q,k,v,causal); tiled=tiled_online_attention(q,k,v,causal,32)
            assert torch.allclose(ref,tiled,atol=1e-5,rtol=1e-4)
            results={}
            for name,fn in (("reference",lambda:reference_attention(q,k,v,causal)),("tiled",lambda:tiled_online_attention(q,k,v,causal,32))):
                for _ in range(2): fn()
                if dev=="cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                ts=[]
                for _ in range(5):
                    if dev=="cuda": torch.cuda.synchronize()
                    t=time.perf_counter(); fn()
                    if dev=="cuda": torch.cuda.synchronize()
                    ts.append(time.perf_counter()-t)
                results[name]={"median_seconds":sorted(ts)[2],"peak_memory_bytes":torch.cuda.max_memory_allocated() if dev=="cuda" else None}
            rows.append({"q_len":qlen,"k_len":klen,"head_dim":d,"causal":causal,"max_abs_error":(ref-tiled).abs().max().item(),"mean_abs_error":(ref-tiled).abs().mean().item(),"timings":results})
    Path("artifacts/results").mkdir(parents=True,exist_ok=True); Path("artifacts/results/attention_benchmark.json").write_text(json.dumps({"device":dev,"warmups":2,"repeats":5,"results":rows},indent=2))
if __name__=="__main__": main()

