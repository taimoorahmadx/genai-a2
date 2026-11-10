"""Interactive model-backed decoding comparison (requires cached/downloadable weights)."""
import time
import streamlit as st
import torch
from trial_screening.modeling import MODEL_ID, load_model
from trial_screening.decoding import greedy_uncached, greedy_cached, speculative_greedy

st.set_page_config(page_title="Inference benchmarks",page_icon="⏱️",layout="wide")
st.title("Inference comparison")
st.warning("Measurements are for this machine and model setup only. No values are prefilled or presented as experimental results.")
model_id=st.text_input("Target model",MODEL_ID)
draft_id=st.text_input("Draft model (must use the same tokenizer)","Qwen/Qwen2.5-0.5B-Instruct")
prompt=st.text_area("Prompt", "Patient has a documented diagnosis. Evaluate the criterion.")
new_tokens=st.slider("Generated tokens",4,96,16)
if st.button("Measure all decoding paths",type="primary"):
    try:
        with st.spinner("Loading models and running warmup plus repeated measurements..."):
            tokenizer,target=load_model(model_id); draft_tokenizer,draft=load_model(draft_id)
            if tokenizer.get_vocab()!=draft_tokenizer.get_vocab(): raise ValueError("Target and draft tokenizers differ")
            device=next(target.parameters()).device
            ids=tokenizer(prompt,return_tensors="pt",truncation=True,max_length=2048).input_ids.to(device)
            funcs={"Uncached reference":lambda:greedy_uncached(target,ids,new_tokens),
                   "KV cached":lambda:greedy_cached(target,ids,new_tokens),
                   "Greedy speculative":lambda:speculative_greedy(target,draft,ids,new_tokens)}
            rows={}
            for name,fn in funcs.items():
                fn(); timings=[]
                if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
                for _ in range(3):
                    if device.type=="cuda": torch.cuda.synchronize(device)
                    start=time.perf_counter(); result=fn()
                    if device.type=="cuda": torch.cuda.synchronize(device)
                    timings.append(time.perf_counter()-start)
                rows[name]={"seconds":sorted(timings)[1],"tokens_per_second":new_tokens/sorted(timings)[1],
                    "peak_memory_mb":torch.cuda.max_memory_allocated(device)/1024**2 if device.type=="cuda" else None,
                    "tokens":result.token_ids[0].tolist(),"accepted":result.accepted_draft_tokens,"proposed":result.proposed_draft_tokens}
        expected=rows["Uncached reference"]["tokens"]
        st.dataframe([{"Mode":k,"Median seconds":round(v["seconds"],4),"Tokens/s":round(v["tokens_per_second"],2),"Peak memory MB":round(v["peak_memory_mb"],1) if v["peak_memory_mb"] is not None else "CPU measurement unavailable","Greedy tokens match":v["tokens"]==expected,"Draft accepted/proposed":f"{v['accepted']}/{v['proposed']}" if "Speculative" in k else "—"} for k,v in rows.items()],use_container_width=True)
        if rows["Greedy speculative"]["tokens"]!=expected: st.error("Speculative output differed from target greedy decoding.")
        else: st.success("All paths produced identical target token IDs for this prompt.")
    except Exception as exc: st.error(f"Benchmark could not run: {exc}")

