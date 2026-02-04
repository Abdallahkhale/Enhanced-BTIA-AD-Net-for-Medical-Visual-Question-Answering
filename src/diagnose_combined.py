
import sys
import yaml
from pathlib import Path
from collections import Counter
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import create_dataloaders

def diagnose_combined():
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Creating combined dataloaders (VQA-RAD + PathVQA + SLAKE)...")
    # Force use of all datasets matching training config
    train_loader, _, train_dataset = create_dataloaders(
        data_dir='data',
        batch_size=32,
        num_workers=0,
        use_slake=config['data'].get('use_slake_augmentation', False),
        use_pathvqa=config['data'].get('use_pathvqa_augmentation', False),
        image_size=224
    )
    
    print(f"\nTotal Training Samples: {len(train_dataset)}")
    
    # Analyze answer types
    type_counts = Counter()
    answer_counts = Counter()
    
    yes_variations = {'yes', 'y'}
    no_variations = {'no', 'n'}
    
    for s in train_dataset.samples:
        ans = s['answer'].lower()
        if ans in yes_variations:
            type_counts['yes'] += 1
        elif ans in no_variations:
            type_counts['no'] += 1
        else:
            type_counts['open'] += 1
            answer_counts[ans] += 1
            
    total = len(train_dataset)
    yes_count = type_counts['yes']
    no_count = type_counts['no']
    open_count = type_counts['open']
    
    print(f"\nDistribution:")
    print(f"  YES:  {yes_count:5d} ({yes_count/total*100:4.1f}%)")
    print(f"  NO:   {no_count:5d} ({no_count/total*100:4.1f}%)")
    print(f"  OPEN: {open_count:5d} ({open_count/total*100:4.1f}%)")
    
    print(f"\nTop 10 Open Answers:")
    for ans, count in answer_counts.most_common(10):
        print(f"  '{ans}': {count}")
        
    print(f"\nUnique Open Answers: {len(answer_counts)}")
    
    # Check if PathVQA is dominating
    sources = Counter(s['source'] for s in train_dataset.samples)
    print(f"\nMix by Source:")
    for src, count in sources.items():
        print(f"  {src}: {count} ({count/total*100:4.1f}%)")

if __name__ == "__main__":
    diagnose_combined()
