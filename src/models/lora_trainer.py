"""
LoRA / QLoRA fine-tuning for Flan-T5-Base on FinQA.

Supports two variants:
- vanilla: fp16 LoRA (no quantization)
- qlora:   4-bit quantized base model + LoRA (memory-efficient)
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
    
    train_ds = Dataset.from_list(train_data)
    dev_ds = Dataset.from_list(dev_data)
    return train_ds, dev_ds


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


def build_model(variant: str, model_name: str = 'google/flan-t5-base'):
    """Build base model + LoRA wrapper for the given variant."""
    if variant == 'vanilla':
        # Standard fp16 LoRA
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
    elif variant == 'qlora':
        # 4-bit quantized base model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
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
    
    # Apply LoRA on attention q, v
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['q', 'v'],
        bias='none',
    )
    model = get_peft_model(model, lora_config)
    
    # Show trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    
    return model


def train_lora(variant: str,
               train_path: str = 'data/processed/training/train.json',
               dev_path: str = 'data/processed/training/dev.json',
               output_dir: str = None,
               model_name: str = 'google/flan-t5-base',
               epochs: int = 3,
               batch_size: int = 16,
               learning_rate: float = 1e-4,
               n_train_samples: int = None):
    """Train a LoRA adapter end-to-end."""
    
    if output_dir is None:
        output_dir = f'/content/drive/MyDrive/finqa-rag-lora-ablation/checkpoints/c3_{variant}'
    
    print(f"=== Training C3 {variant.upper()} LoRA ===")
    print(f"Output dir: {output_dir}")
    
    # Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Data
    print(f"\nLoading data...")
    train_ds, dev_ds = load_data(train_path, dev_path)
    if n_train_samples:
        train_ds = train_ds.select(range(min(n_train_samples, len(train_ds))))
        print(f"  Using {len(train_ds)} training samples (subset)")
    print(f"  Train: {len(train_ds)}, Dev: {len(dev_ds)}")
    
    print("Tokenizing...")
    train_tokenized = tokenize_dataset(train_ds, tokenizer)
    dev_tokenized = tokenize_dataset(dev_ds, tokenizer)
    
    # Model
    print(f"\nBuilding {variant} model...")
    model = build_model(variant, model_name)
    
    # Data collator (handles dynamic padding)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='longest')
    
    # Training arguments
    fp16 = (variant == 'vanilla')  # vanilla uses fp16; qlora uses 4-bit + fp16 compute
    
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=fp16,
        report_to='none',  # disable wandb for now
        predict_with_generate=False,  # eval with loss only (faster)
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        data_collator=collator,
    )
    
    # Train
    print(f"\n=== Starting training ===")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print(f"Steps per epoch: ~{len(train_tokenized) // batch_size}")
    print(f"Total steps:     ~{len(train_tokenized) // batch_size * epochs}")
    
    trainer.train()
    
    # Save final adapter
    final_dir = os.path.join(output_dir, 'final')
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n✅ Saved final adapter to {final_dir}")
    
    return final_dir
