"""
Diagnostic script to analyze training data and identify open-ended issues.
"""

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import MedVQADataset

def diagnose_dataset():
    """Analyze the training dataset for class imbalance issues."""
    
    print("="*60)
    print("Dataset Diagnostic Report")
    print("="*60)
    
    # Load training dataset
    train_dataset = MedVQADataset(
        data_dir="data",
        split="train",
        use_slake=True,
        auto_download=False
    )
    
    # Load test dataset
    test_dataset = MedVQADataset(
        data_dir="data",
        split="test",
        use_slake=False,
        auto_download=False
    )
    
    # Analyze training data
    print("\n--- TRAINING DATA ANALYSIS ---")
    
    closed_count = 0
    open_count = 0
    closed_answers = Counter()
    open_answers = Counter()
    
    for sample in train_dataset.samples:
        answer = sample['answer']
        answer_type = sample.get('answer_type', 'OPEN').upper()
        is_closed = answer_type in ['CLOSED', 'YES/NO'] or answer.lower() in ['yes', 'no']
        
        if is_closed:
            closed_count += 1
            closed_answers[answer] += 1
        else:
            open_count += 1
            open_answers[answer] += 1
    
    print(f"\nTotal training samples: {len(train_dataset.samples)}")
    print(f"Closed-ended samples: {closed_count} ({closed_count/len(train_dataset.samples)*100:.1f}%)")
    print(f"Open-ended samples: {open_count} ({open_count/len(train_dataset.samples)*100:.1f}%)")
    
    print(f"\nUnique closed-ended answers: {len(closed_answers)}")
    print(f"Unique open-ended answers: {len(open_answers)}")
    
    print("\nTop 10 closed-ended answers:")
    for ans, count in closed_answers.most_common(10):
        print(f"  '{ans}': {count}")
    
    print("\nTop 10 open-ended answers:")
    for ans, count in open_answers.most_common(10):
        print(f"  '{ans}': {count}")
    
    # Analyze test data
    print("\n--- TEST DATA ANALYSIS ---")
    
    test_closed_count = 0
    test_open_count = 0
    test_open_answers = Counter()
    
    for sample in test_dataset.samples:
        answer = sample['answer']
        answer_type = sample.get('answer_type', 'OPEN').upper()
        is_closed = answer_type in ['CLOSED', 'YES/NO'] or answer.lower() in ['yes', 'no']
        
        if is_closed:
            test_closed_count += 1
        else:
            test_open_count += 1
            test_open_answers[answer] += 1
    
    print(f"\nTotal test samples: {len(test_dataset.samples)}")
    print(f"Closed-ended samples: {test_closed_count} ({test_closed_count/len(test_dataset.samples)*100:.1f}%)")
    print(f"Open-ended samples: {test_open_count} ({test_open_count/len(test_dataset.samples)*100:.1f}%)")
    
    # Check vocabulary coverage
    print("\n--- VOCABULARY COVERAGE ---")
    
    # Use training vocab
    train_vocab = set(train_dataset.answer_to_idx.keys())
    test_open_in_vocab = sum(1 for ans in test_open_answers if ans in train_vocab)
    
    print(f"Training vocabulary size: {len(train_vocab)}")
    print(f"Test open-ended answers in training vocab: {test_open_in_vocab}/{len(test_open_answers)}")
    
    # Sample of test open-ended answers NOT in training vocab
    missing = [ans for ans in test_open_answers if ans not in train_vocab][:10]
    if missing:
        print(f"\nTest open-ended answers NOT in training vocab (sample):")
        for ans in missing:
            print(f"  '{ans}'")
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    diagnose_dataset()
