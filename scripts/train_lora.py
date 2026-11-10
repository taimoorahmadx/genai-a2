#!/usr/bin/env python3
"""LoRA supervised fine-tuning. Run only on train data; validation selects checkpoint."""
import json, os
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trial_screening.modeling import MODEL_ID, make_prompt
from trial_screening.data import sentence_map

BASE=os.environ.get("BASE_MODEL",MODEL_ID); OUT=os.environ.get("OUTPUT_DIR","artifacts/checkpoints/qwen25-05b-lora")
MAX_LEN=int(os.environ.get("MAX_LENGTH","4096"))

def read(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def main():
    train=read("data/processed/train.jsonl"); valid=read("data/processed/validation.jsonl")
    tokenizer=AutoTokenizer.from_pretrained(BASE); tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
    def encode(rows):
        encoded={"input_ids":[],"attention_mask":[],"labels":[]}
        for r in rows:
            prompt=tokenizer(make_prompt(r),add_special_tokens=False)["input_ids"]
            sentences=sentence_map(r["note"])
            quoted=[f"Sentence {i}: {sentences[i]}" for i in r["evidence_ids"] if i in sentences]
            explanation=("Evidence: " + " ".join(quoted)) if quoted else "No expert evidence sentence is annotated; missing evidence is not proof either way."
            answer=tokenizer("\n"+json.dumps({"label":r["label"],"evidence_ids":r["evidence_ids"],"explanation":explanation})+tokenizer.eos_token,add_special_tokens=False)["input_ids"]
            # Keep answer tokens and the most recent prompt context under MAX_LEN.
            answer=answer[:MAX_LEN-1]; prompt=prompt[-(MAX_LEN-len(answer)):]
            ids=prompt+answer; labels=[-100]*len(prompt)+answer
            encoded["input_ids"].append(ids); encoded["attention_mask"].append([1]*len(ids)); encoded["labels"].append(labels)
        return encoded
    class AnswerOnlyCollator:
        def __call__(self, features):
            width=max(len(x["input_ids"]) for x in features)
            return {key:torch.tensor([x[key]+([(-100 if key=="labels" else 0)]*(width-len(x[key]))) for x in features]) for key in ("input_ids","attention_mask","labels")}
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=["q_proj","k_proj","v_proj","o_proj"],task_type="CAUSAL_LM"))
    args=TrainingArguments(output_dir=OUT,num_train_epochs=3,per_device_train_batch_size=1,per_device_eval_batch_size=1,
      gradient_accumulation_steps=8,learning_rate=2e-4,warmup_ratio=.05,weight_decay=.01,logging_steps=10,
      eval_strategy="epoch",save_strategy="epoch",load_best_model_at_end=True,metric_for_best_model="eval_loss",
      greater_is_better=False,gradient_checkpointing=True,fp16=False,bf16=torch.cuda.is_available(),report_to="none")
    trainer=Trainer(model=model,args=args,train_dataset=Dataset.from_dict(encode(train)),eval_dataset=Dataset.from_dict(encode(valid)),
      data_collator=AnswerOnlyCollator())
    trainer.train(); trainer.save_model(OUT); tokenizer.save_pretrained(OUT)
    Path(OUT,"run_config.json").write_text(json.dumps({"base_model":BASE,"max_length":MAX_LEN,"epochs":3,"learning_rate":2e-4,"batch":1,"gradient_accumulation":8,"selection":"validation loss","test_used":False},indent=2))
if __name__=="__main__": main()

