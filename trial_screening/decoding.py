"""Explicit greedy generation and speculative decoding for HF causal LMs."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch

@dataclass
class DecodeResult:
    token_ids: torch.Tensor
    accepted_draft_tokens: int=0
    proposed_draft_tokens: int=0

def _next(out: Any) -> torch.Tensor: return out.logits[:,-1,:].argmax(-1,keepdim=True)
def _past_len(cache: Any) -> int:
    return cache.get_seq_length() if hasattr(cache,"get_seq_length") else cache[0][0].shape[-2]

@torch.inference_mode()
def greedy_uncached(model: Any, input_ids: torch.Tensor, max_new_tokens: int) -> DecodeResult:
    seq=input_ids
    for _ in range(max_new_tokens): seq=torch.cat([seq,_next(model(input_ids=seq,use_cache=False))],dim=-1)
    return DecodeResult(seq[:,input_ids.shape[-1]:])

@torch.inference_mode()
def greedy_cached(model: Any, input_ids: torch.Tensor, max_new_tokens: int) -> DecodeResult:
    out=model(input_ids=input_ids,use_cache=True); cache=out.past_key_values; tok=_next(out); generated=[]
    for i in range(max_new_tokens):
        generated.append(tok)
        if i+1==max_new_tokens: break
        n=_past_len(cache); mask=torch.ones((input_ids.shape[0],n+1),dtype=torch.long,device=input_ids.device)
        pos=torch.full((input_ids.shape[0],1),n,dtype=torch.long,device=input_ids.device)
        out=model(input_ids=tok,past_key_values=cache,attention_mask=mask,position_ids=pos,use_cache=True)
        cache=out.past_key_values; tok=_next(out)
    return DecodeResult(torch.cat(generated,dim=-1) if generated else input_ids[:,:0])

@torch.inference_mode()
def speculative_greedy(target: Any, draft: Any, input_ids: torch.Tensor, max_new_tokens: int, draft_tokens: int=4) -> DecodeResult:
    """Draft tokens, verify with target, append matching prefix and correction.

    Uses same-tokenizer models. Caches are re-prefilled after rejection so their
    state always exactly represents the emitted sequence, at some extra cost.
    """
    actual=input_ids; accepted=proposed=0; emitted=[]
    while len(emitted)<max_new_tokens:
        ot=target(input_ids=actual,use_cache=True); od=draft(input_ids=actual,use_cache=True)
        tc,dc=ot.past_key_values,od.past_key_values
        target_next,draft_next=_next(ot),_next(od); draft_seq=[]
        for _ in range(min(draft_tokens,max_new_tokens-len(emitted))):
            draft_seq.append(draft_next)
            od=draft(input_ids=draft_next,past_key_values=dc,use_cache=True); dc=od.past_key_values; draft_next=_next(od)
        cand=torch.cat(draft_seq,dim=-1); proposed += cand.shape[-1]
        n=_past_len(tc); verify=target(input_ids=cand,past_key_values=tc,
            attention_mask=torch.ones((cand.shape[0],n+cand.shape[-1]),dtype=torch.long,device=cand.device),
            position_ids=torch.arange(n,n+cand.shape[-1],device=cand.device).expand(cand.shape[0],-1),use_cache=True)
        expected=torch.cat([target_next,verify.logits[:,:-1,:].argmax(-1)],dim=1)
        mismatch=torch.nonzero(expected[0]!=cand[0],as_tuple=False)
        count=int(mismatch[0,0]) if mismatch.numel() else cand.shape[-1]
        accepted += count
        emitted.extend(cand[:,i:i+1] for i in range(count)); actual=torch.cat([input_ids]+emitted,dim=-1)
        if count<cand.shape[-1] and len(emitted)<max_new_tokens:
            emitted.append(expected[:,count:count+1]); actual=torch.cat([input_ids]+emitted,dim=-1)
    return DecodeResult(torch.cat(emitted[:max_new_tokens],dim=-1),accepted,proposed)

