"""Debug what predictions the model is making for open-ended questions."""

import sys
sys.path.insert(0, '.')

import torch
from collections import Counter
from data.dataset import create_dataloaders
import yaml
from pathlib import Path
import yaml
from pathlib import Path


def analyze_test_distribution():
    """Analyze test set distribution and model predictions."""
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, test_loader, train_dataset = create_dataloaders(
        data_dir='data',
        batch_size=8,
        num_workers=0,
        use_slake=False,
        use_pathvqa=True,
        image_size=224
    )
    
    print(f"Num answers: {train_dataset.num_answers}")
    
    # Analyze targets
    closed_count = 0
    open_count = 0
    open_targets = []
    closed_targets = []
    
    for batch in test_loader:
        is_closed = batch['is_closed']
        targets = batch['answer_idx']
        
        for i in range(len(is_closed)):
            if is_closed[i]:
                closed_count += 1
                closed_targets.append(targets[i].item())
            else:
                open_count += 1
                open_targets.append(targets[i].item())
    
    print(f"\nTest set: {closed_count} CLOSED, {open_count} OPEN")
    
    # Check closed target distribution
    closed_dist = Counter(closed_targets)
    print(f"\nCLOSED target distribution:")
    for idx, count in closed_dist.most_common(5):
        ans = train_dataset.idx_to_answer.get(idx, 'UNK')
        print(f"  idx={idx}, ans='{ans}', count={count}")
    
    # Check open target distribution
    open_dist = Counter(open_targets)
    print(f"\nOPEN target distribution ({len(open_dist)} unique answers):")
    for idx, count in open_dist.most_common(10):
        ans = train_dataset.idx_to_answer.get(idx, 'UNK')
        print(f"  idx={idx}, ans='{ans[:40]}', count={count}")
    
    # Check if most common open answer is 'yes' or 'no' (classification bug)
    print("\n--- CHECKING FOR CLASSIFICATION BUGS ---")
    
    # Find yes/no indices
    yes_idx = train_dataset.answer_to_idx.get('yes', -1)
    no_idx = train_dataset.answer_to_idx.get('no', -1)
    
    print(f"'yes' index: {yes_idx}")
    print(f"'no' index: {no_idx}")
    
    # Check if open targets include yes/no
    open_yes_no = sum(1 for t in open_targets if t in [yes_idx, no_idx])
    print(f"\nOPEN targets that are yes/no: {open_yes_no}/{len(open_targets)}")
    
    if open_yes_no > 0:
        print("WARNING: Some OPEN targets are yes/no - this is a data labeling issue!")
    
    # Load checkpoint if exists and check predictions
    ckpt_path = Path('checkpoints/best_model.pt')
    if ckpt_path.exists():
        print("\n--- CHECKPOINT EXISTS - Manual analysis recommended ---")
        print("Run: python src/train.py --eval to see predictions")
    else:
        print("\nNo checkpoint found at checkpoints/best_model.pt")


if __name__ == "__main__":
    analyze_test_distribution()
