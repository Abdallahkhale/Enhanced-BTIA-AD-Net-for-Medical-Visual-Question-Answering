
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import MedVQADataset

def verify_weights():
    print("Loading dataset...")
    ds = MedVQADataset(data_dir="data", split="train", use_slake=True, auto_download=False)
    
    print("Calculating weights...")
    weights = ds.get_class_weights()
    
    yes_idx = ds.answer_to_idx['yes']
    no_idx = ds.answer_to_idx['no']
    rare_idx = ds.answer_to_idx['axial']
    
    print(f"\nWeight for 'yes' (count={ds._answer_counts['yes']}): {weights[yes_idx]:.4f}")
    print(f"Weight for 'no'  (count={ds._answer_counts['no']}): {weights[no_idx]:.4f}")
    print(f"Weight for 'axial' (count={ds._answer_counts['axial']}): {weights[rare_idx]:.4f}")
    
    ratio = weights[rare_idx] / weights[yes_idx]
    print(f"\nRatio rare/frequent: {ratio:.1f}x")
    
    if ratio > 10:
        print("SUCCESS: Rare classes are significantly upweighted.")
    else:
        print("FAILURE: Weights are not aggressive enough.")

if __name__ == "__main__":
    verify_weights()
