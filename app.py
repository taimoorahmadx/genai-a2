"""Streamlit browser UI for local criterion screening."""
import json
import streamlit as st
from trial_screening.modeling import MODEL_ID, load_model, predict
from trial_screening.data import sentence_map

st.set_page_config(page_title="Trial Criterion Screening", page_icon="🧪", layout="wide")
st.title("Clinical Trial Criterion Screening")
st.warning("Educational screening prototype only. This is not a clinical enrollment decision.")
st.caption("Each criterion is assessed separately. Missing information is not evidence of a negative finding.")
with st.sidebar:
    model_id=st.text_input("Open-weight model",MODEL_ID)
    adapter=st.text_input("Optional local LoRA adapter path","")
    st.markdown("**Inference comparison**")
    st.write("The Benchmarks page reports measured reference, cached and speculative decoding performance when run locally.")
note=st.text_area("Patient note",height=220,placeholder="Enter numbered patient sentences, for example: 0. Patient is 58 years old. 1. ...")
criteria=st.text_area("Trial criteria, one per line; prefix with inclusion: or exclusion:",height=160,placeholder="inclusion: Confirmed diagnosis of ...\nexclusion: Prior treatment with ...")
if st.button("Run screening",type="primary"):
    if not note.strip() or not criteria.strip(): st.error("Enter a patient note and at least one criterion.")
    else:
        try:
            with st.spinner("Loading local model and screening criteria..."):
                tokenizer,model=load_model(model_id,adapter or None)
                rows=[]
                for line in criteria.splitlines():
                    if not line.strip(): continue
                    kind="inclusion"
                    if ":" in line and line.split(":",1)[0].strip().lower() in {"inclusion","exclusion"}:
                        kind,text=line.split(":",1); kind=kind.strip().lower(); text=text.strip()
                    else: text=line.strip()
                    p=predict(tokenizer,model,{"criterion_type":kind,"criterion_text":text,"note":note})
                    rows.append({"criterion_type":kind,"criterion":text,**p})
            st.subheader("Criterion level results")
            for i,r in enumerate(rows,1):
                with st.expander(f"{i}. {r['criterion_type'].title()}: {r['criterion']} — {r['label']}",expanded=True):
                    st.write(r["explanation"] or "No explanation returned.")
                    st.write("Evidence sentence IDs:",r["evidence_ids"] or "None cited")
                    for sid,text in r["evidence_text"].items(): st.markdown(f"**Sentence {sid}:** {text}")
            if rows:
                labels=[r["label"] for r in rows]
                if any(x=="excluded" for x in labels): overall="Potential exclusion criterion identified; requires expert review."
                elif any(x=="not enough information" for x in labels): overall="Insufficient information for a confident preliminary assessment."
                elif all(x in {"included","not excluded","not applicable"} for x in labels): overall="No disqualifying criterion detected by this prototype; expert review is still required."
                else: overall="Mixed criterion findings; expert review required."
                st.info("Rule-based aggregation (not a dataset ground-truth label): "+overall)
                st.download_button("Download readable JSON report",json.dumps({"notice":"Educational prototype; not a clinical decision","model":model_id,"criteria":rows,"aggregation":overall},indent=2),file_name="screening_report.json",mime="application/json")
        except Exception as e: st.error(f"Could not run screening: {e}")

