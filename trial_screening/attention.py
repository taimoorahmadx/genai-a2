"""Reference and tiled online-softmax attention implementations."""
from __future__ import annotations
import math
import torch

def _causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    qp = torch.arange(k_len-q_len, k_len, device=device)[:, None]
    kp = torch.arange(k_len, device=device)[None, :]
    return kp <= qp

def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool=False, mask: torch.Tensor|None=None) -> torch.Tensor:
    scores = q @ k.transpose(-2,-1) / math.sqrt(q.shape[-1])
    if causal: scores = scores.masked_fill(~_causal_mask(q.shape[-2], k.shape[-2], q.device), torch.finfo(scores.dtype).min)
    if mask is not None: scores = scores.masked_fill(~mask.to(torch.bool), torch.finfo(scores.dtype).min)
    return torch.softmax(scores.float(), dim=-1).to(q.dtype) @ v

def tiled_online_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool=False, block_size: int=64, mask: torch.Tensor|None=None) -> torch.Tensor:
    """Exact query/key blocked attention with running maxima, sums and weighted values."""
    if block_size < 1: raise ValueError("block_size must be positive")
    b,h,nq,d = q.shape; nk = k.shape[-2]; scale = 1/math.sqrt(d); chunks=[]
    for qs in range(0,nq,block_size):
        qe=min(qs+block_size,nq); qb=q[...,qs:qe,:].float()
        m=torch.full((b,h,qe-qs,1),-torch.inf,device=q.device); l=torch.zeros_like(m)
        acc=torch.zeros((b,h,qe-qs,v.shape[-1]),device=q.device,dtype=torch.float32)
        for ks in range(0,nk,block_size):
            ke=min(ks+block_size,nk); s=qb @ k[...,ks:ke,:].float().transpose(-2,-1)*scale
            allowed=torch.ones((qe-qs,ke-ks),dtype=torch.bool,device=q.device)
            if causal:
                qi=torch.arange(nk-nq+qs,nk-nq+qe,device=q.device)[:,None]; ki=torch.arange(ks,ke,device=q.device)[None,:]
                allowed &= ki <= qi
            if mask is not None: allowed &= mask[...,qs:qe,ks:ke].to(torch.bool)
            s=s.masked_fill(~allowed,-torch.inf); bm=s.amax(-1,keepdim=True); nm=torch.maximum(m,bm)
            alpha=torch.where(torch.isfinite(m),torch.exp(m-nm),0.)
            p=torch.where(torch.isfinite(s),torch.exp(s-nm),0.)
            l=l*alpha+p.sum(-1,keepdim=True); acc=acc*alpha+p @ v[...,ks:ke,:].float(); m=nm
        chunks.append((acc/l.clamp_min(torch.finfo(torch.float32).tiny)).to(q.dtype))
    return torch.cat(chunks,dim=-2)

