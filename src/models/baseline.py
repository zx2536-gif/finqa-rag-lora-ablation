"""
Flan-T5-Base zero-shot baseline model wrapper.

Loads the model once, exposes a `predict_batch` method for inference.
"""

from typing import List
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class FlanT5Baseline:
    """Zero-shot inference wrapper for Flan-T5-Base."""
    
    def __init__(self, model_name: str = "google-t5/t5-base", 
                 device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        print(f"Loading {model_name} on {device}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        print(f"Model loaded. Total params: {sum(p.numel() for p in self.model.parameters()):,}")
    
    @torch.no_grad()
    def predict_batch(self, prompts: List[str], 
                      max_input_length: int = 512,
                      max_new_tokens: int = 64,
                      batch_size: int = 8) -> List[str]:
        """Generate predictions for a list of prompts.
        
        Args:
            prompts: list of input strings
            max_input_length: truncate prompts to this many tokens
            max_new_tokens: max tokens in generated answer
            batch_size: how many prompts to process per batch
        
        Returns:
            List of decoded prediction strings.
        """
        predictions = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic
                num_beams=1,
            )
            
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded)
        
        return predictions
