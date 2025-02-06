import os
import torch
from transformers import pipeline


def unsloth_model(args):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel(
        model_name="google/gemma-2-9b-it",
        load_in_4bit=True,
        max_seq_length=2048
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.dropout   #과적합 방지, 일반화 성능 향상
    )
    return model, tokenizer


def generation_pipe(model_id):
    pipe = pipeline(
        task="text-generation",
        model=model_id,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        token=os.environ.get("HF_TOKEN"),
        model_kwargs={"max_length": 2048}
        )
    return pipe