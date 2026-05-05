"""
C3: LoRA-only inference pipeline.

Loads a trained LoRA adapter and runs inference on finqa_500 evaluation set.
Same context format and metrics as C1 baseline (no retrieval, no RAG).

Usage:
    python -m src.pipelines.c3_lora --variant vanilla
    python -m src.pipelines.c3_lora --variant qlora
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.utils.data_utils import build_prompt, get_gold_answer
from src.evaluation.metrics import evaluate_predictions


class FlanT5LoRA:
    """Inference wrapper that loads a base T5 + LoRA adapter."""
    
    def __init__(self, adapter_dir: str,
                 base_model: str = 'google/flan-t5-base',
                 variant: str = 'vanilla'):
        from transformers import T5ForConditionalGeneration, T5Tokenizer, BitsAndBytesConfig
        from peft import PeftModel
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading base model: {base_model} ({variant})")
        
        if variant == 'qlora':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            base = T5ForConditionalGeneration.from_pretrained(
                base_model, quantization_config=bnb_config, device_map='auto'
            )
        else:
            base = T5ForConditionalGeneration.from_pretrained(
                base_model, torch_dtype=torch.float16
            ).to(self.device)
        
        print(f"Loading LoRA adapter from {adapter_dir}")
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.eval()
        
        self.tokenizer = T5Tokenizer.from_pretrained(adapter_dir)
        print("Model ready.")
    
    @torch.no_grad()
    def predict_batch(self, prompts, max_input_length=512, max_new_tokens=64, batch_size=16):
        predictions = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i+batch_size]
            inputs = self.tokenizer(
                batch, return_tensors='pt', padding=True, truncation=True,
                max_length=max_input_length,
            ).to(self.device)
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            )
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded)
        return predictions


def run_c3(adapter_dir: str, variant: str,
           data_path: str = 'data/processed/finqa_500.json',
           out_dir: str = 'results/metrics',
           model_name: str = 'google/flan-t5-base',
           batch_size: int = 16):
    
    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        samples = json.load(f)
    print(f"  {len(samples)} samples")
    
    print("\nBuilding prompts...")
    prompts = [build_prompt(s) for s in samples]
    golds = [get_gold_answer(s) for s in samples]
    strata = [s['_stratum'] for s in samples]
    
    model = FlanT5LoRA(adapter_dir, base_model=model_name, variant=variant)
    
    print(f"\nGenerating predictions (batch_size={batch_size})...")
    predictions = model.predict_batch(prompts, batch_size=batch_size)
    
    print("\nEvaluating...")
    results = evaluate_predictions(predictions, golds, strata)
    
    print(f"\n=== C3 {variant.upper()} LoRA Results ===")
    print(f"Overall ({results['overall']['n_samples']} samples):")
    print(f"  F1:      {results['overall']['f1']:.4f}")
    print(f"  ROUGE-L: {results['overall']['rouge_l']:.4f}")
    print("\nPer-stratum:")
    for st in sorted(results['by_stratum'].keys()):
        m = results['by_stratum'][st]
        print(f"  {st:25s} n={m['n_samples']:3d}  "
              f"F1={m['f1']:.4f}  ROUGE-L={m['rouge_l']:.4f}")
    
    # Save
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f'c3_{variant}'
    
    metrics_path = os.path.join(out_dir, f'{tag}_metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'config': f'C3_LoRA_{variant}',
            'variant': variant,
            'adapter_dir': adapter_dir,
            'model': model_name,
            'data_path': data_path,
            'n_samples': len(samples),
            'timestamp': timestamp,
            'metrics': results,
        }, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    
    preds_path = os.path.join(out_dir, f'{tag}_predictions_{timestamp}.json')
    with open(preds_path, 'w') as f:
        records = [{
            'id': samples[i].get('id', f'sample_{i}'),
            'stratum': strata[i],
            'question': samples[i]['qa']['question'],
            'gold': golds[i],
            'prediction': predictions[i],
        } for i in range(len(samples))]
        json.dump(records, f, indent=2)
    print(f"Saved predictions to {preds_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['vanilla', 'qlora'], required=True)
    parser.add_argument('--adapter_dir', default=None,
                        help='Default: checkpoints/c3_{variant}/final')
    parser.add_argument('--data', default='data/processed/finqa_500.json')
    parser.add_argument('--out_dir', default='results/metrics')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    if args.adapter_dir is None:
        args.adap# 在 Colab 直接调用 train_lora 函数（不通过命令行）
import sys
sys.path.insert(0, '/content/drive/MyDrive/finqa-rag-lora-ablation')

# 重新导入（如果之前 import 过）
import importlib
import src.models.lora_trainer
importlib.reload(src.models.lora_trainer)
from src.models.lora_trainer import train_lora

# Quick test: 用 200 条样本 + 1 epoch + 小 batch
quick_dir = train_lora(
    variant='vanilla',
    output_dir='/tmp/c3_vanilla_quicktest',
    epochs=1,
    batch_size=8,
    n_train_samples=200,
)
print(f"\n✅ Quick test done. Adapter saved to {quick_dir}")ter_dir = f'/content/drive/MyDrive/finqa-rag-lora-ablation/checkpoints/c3_{args.variant}/final'
    
    run_c3(args.adapter_dir, args.variant, args.data, args.out_dir,
           batch_size=args.batch_size)
