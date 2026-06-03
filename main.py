"""Fine-tune google/gemma-2b on a Spanish dataset with QLoRA (4-bit) and TRL's SFTTrainer.

Loads the gated Gemma-2B base in 4-bit (bitsandbytes nf4), attaches a LoRA adapter, trains a
short run on ``jhonparra18/spanish_billion_words_clean`` and pushes the result to the Hugging
Face Hub. Requires a CUDA GPU and a Hugging Face token (with Gemma access granted) in a local
``.env`` file as ``HUGGINGFACE_API_KEY``. The ``# %%`` markers delimit cells for interactive
(Jupyter/IDE) execution; run end-to-end with ``python main.py``.
"""
import os

# Restrict training to GPUs 0 and 1 (set before importing torch). Adjust to your hardware.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from dotenv import load_dotenv, find_dotenv

# %%

load_dotenv(find_dotenv())

HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
model_name = "google/gemma-2b"

# %%

dataset = load_dataset("jhonparra18/spanish_billion_words_clean",
                       trust_remote_code=True,
                       download_mode="reuse_cache_if_exists")

# %%

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, return_tensors="pt", token=HUGGINGFACE_API_KEY)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map='auto', token=HUGGINGFACE_API_KEY)

# %%

def tokenize_function(example):
    return tokenizer(example["text"])


dataset = dataset['train'].train_test_split(test_size=0.1)

# %%

# input_text = "What should I do on a trip to Europe?"
#
# input_ids = tokenizer(input_text, return_tensors="pt")
# outputs = model.generate(**input_ids, max_length=128)
# print(tokenizer.decode(outputs[0]))


# %%

data = dataset.map(tokenize_function, batched=True)

# %%

train_data = data["train"]
test_data = data["test"]

# %%

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# %%

def formatting_func(example):
    text = example['text'][0]
    return [text]


trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=1,
        max_steps=60,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=1,
        output_dir="gemma-2b-spanishbillionwords",
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        save_total_limit=2,
        load_best_model_at_end=True,
        save_steps=1,
        evaluation_strategy='steps',
        save_strategy='steps',
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
    # tokenizer=tokenizer,
)
trainer.train()

# %%

trainer.push_to_hub("mcamara/gemma-2b-es-spanishbillionwords")
