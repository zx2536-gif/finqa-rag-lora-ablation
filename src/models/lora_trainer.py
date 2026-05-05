"""
LoRA / QLoRA fine-tuning for Flan-T5-Base on FinQA.

Supports two variants:
- vanilla: bf16 LoRA (no quantization, A100-friendly)
- qlora:   4-bit quantized base model + LoRA (memory-efficient)

Hyperparameters tuned to avoid mode collapse: higher LR (3e-4), more
epochs (5), larger LoRA rank (16), full attention coverage (q,k,v,o).
"""

import json
import os
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset


def load_data(train_path: str, dev_path: str):
    """Load formatted train/dev pairs as HuggingFace Datasets."""
    with open(train_path) as f:
        train_data = json.load(f)
    with open(dev_path) as f:
        dev_data = json.load(f)
    return Dataset.from_list(train_data), Dataset.from_list(dev_data)


def tokenize_dataset(dataset, tokenizer, max_input_length=512, max_target_length=64):
    """Tokenize input/target pairs for seq2seq training."""
    def _tokenize(batch):
        model_inputs = tokenizer(
            batch['input'],
            max_length=max_input_length,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch['target'],
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    return dataset.map(_tokenize, batched=True, remove_columns=['id', 'input', 'target'])


def build_model(variant: str,
                model_name: str = 'google-t5/t5-base',
                lora_r: int = 16,
                lora_alpha: int = 32,
                lora_dropout: float = 0.1,
                target_modules=None):
    """Build base model + LoRA wrapper.
    
    Default target_modules covers the full attention block (q, k, v, o)
    rather than just q/v -- this gives LoRA enough capacity to learn 
    real reasoning, not just output bias.
    """
    if target_modules is None:
        target_modules = ['q', 'k', 'v', 'o']
    
    if variant == 'vanilla':
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
    elif variant == 'qlora':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map='auto',
        )
        model = prepare_model_for_kbit_training(model)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias='none',
    )
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA config: r={lora_r}, alpha={lora_alpha}, "
          f"target_modules={target_modules}")
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({trainable/total*100:.2f}%)")
    
    return model


def train_lora(variant: str,
               train_path: str = 'data/processed/training/train.json',
               dev_path: str = 'data/processed/training/dev.json',
               output_dir: str = None,
               model_name: str = 'google-t5/t5-base',
               epochs: int = 5,
               batch_size: int = 16,
               learning_rate: float = 3e-4,
               warmup_ratio: float = 0.1,
               lora_r: int = 16,
               lora_alpha: int = 32,
               n_train_samples: int = None):
    """Train a LoRA adapter end-to-end.
    
    Default hyperparameters are tuned for FinQA + Flan-T5-Base based on 
    initial mode-collapse experiment (LR=1e-4 was insufficient).
    """
    if output_dir is None:
        output_dir = f'/content/drive/MyDrive/finqa-rag-lora-ablation/checkpoints/c3_{variant}'
    
    print(f"=== Training C3 {variant.upper()} LoRA ===")
    print(f"Output dir: {output_dir}")
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    print(f"\nLoading data...")
    train_ds, dev_ds = load_data(train_path, dev_path)
    if n_train_samples:
        train_ds = train_ds.select(range(min(n_train_samples, len(train_ds))))
        print(f"  Using {len(train_ds)} training samples (subset)")
    print(f"  Train: {len(train_ds)}, Dev: {len(dev_ds)}")
    
    print("Tokenizing...")
    train_tokenized = tokenize_dataset(train_ds, tokenizer)
    dev_tokenized = tokenize_dataset(dev_ds, tokenizer)
    
    print(f"\nBuilding {variant} model...")
    model = build_model(variant, model_name, lora_r=lora_r, lora_alpha=lora_alpha)
    
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='longest')
    
    use_bf16 = (variant == 'vanilla')
    
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type='cosine',
        logging_steps=20,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        bf16=use_bf16,
        report_to='none',
        predict_with_generate=False,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        data_collator=collator,
    )
    
    print(f"\n=== Starting training ===")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}, "
          f"Warmup: {warmup_ratio}, Scheduler: cosine")
    print(f"Steps per epoch: ~{len(train_tokenized) // batch_size}")
    print(f"Total steps:     ~{len(train_tokenized) // batch_size * epochs}")
    
    trainer.train()
    
    final_dir = os.path.join(output_dir, 'final')
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n✅ Saved final adapter to {final_dir}")
    
    return final_dir
